// ============================================================================
// Forward Boundary Element Method (BEM) model using the rosetta-generated
// `arch3` WebAssembly (embind) binding — the wasm counterpart of
// example_python.py.
//
//       Model -> Surface (fault) -> boundary conditions -> remote stress
//             -> iterative solve -> post-process displacement & stress
//
// Build the module first (needs emscripten + a C++26 / p2996 toolchain):
//     cd bindings/wasm-expanded && emcmake cmake -B build && cmake --build build
// which produces bindings/wasm-expanded/build/arch3.js (+ arch3.wasm). The
// module is MODULARIZE'd with EXPORT_NAME=createModule.
//
// ---------------------------------------------------------------------------
// DIFFERENCES vs example_python.py (embind idioms + backend support):
//   - The module loads asynchronously: `await createModule()`.
//   - std::vector args/returns are embind vector objects: build with
//     `new Module.vector_double()` + push_back; read with .size()/.get();
//     free with .delete().
//   - Callbacks ARE marshalled: the expanded backend wraps a JS function into
//     the underlying std::function via an emscripten::val adapter, so — like
//     example_python.py — this uses `new Module.UserRemote(fn, true)` for a
//     depth-varying far-field stress and `solver.onMessage`/`onEnd` for progress.
//     UserRemote reaching addRemote relies on the emitted
//     `emscripten::base<BaseRemote>` inheritance.
//
// Raw pointers to bound classes ARE marshalled (via emscripten::allow_raw_pointers),
// so Model.addSurface(...) -> Surface*, Model.addRemote(BaseRemote*) and
// Postprocess.burgersFor(Surface*) all work — a returned pointer becomes a
// non-owning JS handle to the C++ object.
// ============================================================================

"use strict";

const path = require("path");
const createModule = require(path.join(__dirname, "bindings", "wasm-expanded", "build", "arch3.js"));

// Build an embind vector_<T> from a JS array (caller must .delete() it).
function toVec(Module, ctorName, arr) {
    const v = new Module[ctorName]();
    for (const x of arr) v.push_back(x);
    return v;
}
// Copy an embind vector into a JS array (does not delete the source).
function fromVec(v) {
    const out = new Array(v.size());
    for (let i = 0; i < out.length; ++i) out[i] = v.get(i);
    return out;
}

// Flat rectangular triangulated patch (the fault surface).
function makeStripMesh(origin, axis1, axis2, n1, n2) {
    const positions = [];
    for (let j = 0; j <= n2; ++j) {
        for (let i = 0; i <= n1; ++i) {
            for (let k = 0; k < 3; ++k) {
                positions.push(origin[k] + (axis1[k] * i) / n1 + (axis2[k] * j) / n2);
            }
        }
    }
    const indices = [];
    for (let j = 0; j < n2; ++j) {
        for (let i = 0; i < n1; ++i) {
            const v0 = j * (n1 + 1) + i;
            const v1 = v0 + 1;
            const v2 = v0 + (n1 + 1);
            const v3 = v2 + 1;
            indices.push(v0, v1, v2, v1, v3, v2);
        }
    }
    return { positions, indices };
}

(async () => {
    const Module = await createModule();

    // ------------------------------------------------------------------------
    // 1. Model + material
    // ------------------------------------------------------------------------
    const model = new Module.Model();
    model.setHalfSpace(false);
    model.setMaterial(0.25, 20e9, 2500); // poisson, young [Pa], density

    // ------------------------------------------------------------------------
    // 2. Fault surface (the discontinuity)
    // ------------------------------------------------------------------------
    const { positions, indices } = makeStripMesh(
        [-1000.0, 0.0, 0.0],   // origin
        [2000.0, 0.0, 0.0],    // strike (x)
        [0.0, 0.0, -1000.0],   // down-dip (-z)
        20,
        10
    );

    const posVec = toVec(Module, "vector_double", positions);
    const idxVec = toVec(Module, "vector_int", indices);
    const fault = model.addSurface(posVec, idxVec);
    posVec.delete();
    idxVec.delete();

    fault.setBcType("dip", "free");     // allow dip-slip
    fault.setBcType("strike", "free");  // allow strike-slip
    fault.setBcType("normal", "fixed"); // no fault opening/closing

    console.log(`Fault: ${fault.nbTriangles()} triangles, ${fault.nbVertices()} vertices`);
    console.log(`Model dof: ${model.nbDof()}`);

    // ------------------------------------------------------------------------
    // 3. Far-field (remote) stress
    // ------------------------------------------------------------------------
    // Background stress state applied at infinity, returned flat as
    // [Sxx, Sxy, Sxz, Syy, Syz, Szz] (Pa). The fault plane is x-z (normal along
    // y), so its traction is sigma . [0,1,0] = [Sxy, Syy, Syz]: the Sxy term is
    // the shear that drives strike-slip. UserRemote(fn, true) wraps this JS
    // callback into the C++ std::function and evaluates it per point (depth-varying).
    function remoteStress(x, y, z) {
        const Sv = -2500.0 * 9.81 * Math.abs(z); // vertical, compressive with depth
        const SHmax = 0.8 * Sv;
        const Shmin = 0.5 * Sv;
        const Sxy = 0.4 * Sv;                     // shear on the fault -> strike-slip
        return [Shmin, Sxy, 0.0, SHmax, 0.0, Sv];
    }

    const remote = new Module.UserRemote(remoteStress, true); // true => the field is a stress
    model.addRemote(remote);

    // ------------------------------------------------------------------------
    // 4. Solve (Gauss-Seidel iterative BEM solver)
    // ------------------------------------------------------------------------
    const solver = new Module.SeidelSolver(model);
    solver.setEps(1e-9);
    solver.setMaxIter(200);
    solver.onMessage((msg) => console.log(`  [solver] ${msg}`));
    solver.onEnd(() => console.log("  [solver] done"));

    console.log("\n--- Solving ---");
    const ok = solver.run(true);
    console.log(`Solver converged: ${ok}`);

    // ------------------------------------------------------------------------
    // 5. Post-process: slip on the fault, and field in the medium
    // ------------------------------------------------------------------------
    const post = new Module.Postprocess(model);

    const burgersVec = post.burgersFor(fault, true, true); // local, atTriangles
    const burgers = fromVec(burgersVec);
    burgersVec.delete();
    const slipMag = [];
    for (let i = 0; i < burgers.length; i += 3) {
        slipMag.push(Math.hypot(burgers[i], burgers[i + 1], burgers[i + 2]));
    }
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    console.log("\n--- Slip on fault ---");
    console.log(`  max |slip|  = ${Math.max(...slipMag).toExponential(4)} m`);
    console.log(`  mean |slip| = ${mean(slipMag).toExponential(4)} m`);

    // Field at observation points OFF the fault plane (y != 0).
    const obs = [];
    for (let k = 0; k < 11; ++k) obs.push(0.0, 200.0, -100.0 * k);
    const obsVec = toVec(Module, "vector_double", obs);
    const dispVec = post.displ(obsVec);   // flat [ux,uy,uz, ...]
    const stressVec = post.stress(obsVec); // flat [Sxx,Sxy,Sxz,Syy,Syz,Szz, ...]
    obsVec.delete();
    const disp = fromVec(dispVec);
    const stress = fromVec(stressVec);
    dispVec.delete();
    stressVec.delete();

    console.log("\n--- Field at observation points (x=0, y=200) ---");
    console.log(`  ${"z (m)".padStart(8)} ${"|u| (m)".padStart(12)} ${"Szz (Pa)".padStart(14)}`);
    for (let k = 0; k < obs.length / 3; ++k) {
        const z = obs[k * 3 + 2];
        const umag = Math.hypot(disp[k * 3], disp[k * 3 + 1], disp[k * 3 + 2]);
        const szz = stress[k * 6 + 5];
        console.log(
            `  ${z.toFixed(0).padStart(8)} ${umag.toExponential(4).padStart(12)} ${szz.toExponential(4).padStart(14)}`
        );
    }

    console.log("\nDone.");
})();
