// ============================================================================
// Forward Boundary Element Method (BEM) model using the rosetta-generated
// `arch3` Node (N-API) binding — the node counterpart of example_python.py.
//
//       Model -> Surface (fault) -> boundary conditions -> remote stress
//             -> iterative solve -> post-process displacement & stress
//
// Build the addon first (needs Node + a C++26 / p2996 toolchain):
//     cd bindings/node && npm install && npm run build
// which produces bindings/node/build/Release/arch3.node. Then: node example_node.js
//
// ---------------------------------------------------------------------------
// This runs: the Node backend marshals raw pointers to bound classes, so
// Model.addSurface(...) -> Surface*, Model.addRemote(BaseRemote*) and
// Postprocess.burgersFor(Surface*) all work (a returned pointer becomes a
// non-owning JS handle to the C++ object).
//
// Like example_python.py, this uses the callback forms: `UserRemote(fn, true)`
// takes a JS function evaluated per point to give a depth-varying far-field
// stress, and `solver.onMessage` / `solver.onEnd` receive progress callbacks.
// The Node backend marshals a JS function into the underlying std::function
// parameter (the callback runs synchronously on the JS thread during the
// solve), so this is a faithful counterpart of the python script.
// ============================================================================

"use strict";

const path = require("path");
const arch3 = require(path.join(__dirname, "bindings", "node", "build", "Release", "arch3.node"));

// ---------------------------------------------------------------------------
// Helper: build a flat rectangular triangulated patch (the fault surface)
//   positions = [x0,y0,z0, x1,y1,z1, ...]   indices = [i0,i1,i2, ...]
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 1. Model + material
// ---------------------------------------------------------------------------
const model = new arch3.Model();
model.setHalfSpace(false);
model.setMaterial(0.25, 20e9, 2500); // poisson, young [Pa], density

// ---------------------------------------------------------------------------
// 2. Fault surface (the discontinuity)
// ---------------------------------------------------------------------------
// A vertical fault, 2 km along strike (x) x 1 km down dip (-z), centred at y=0.
const { positions, indices } = makeStripMesh(
    [-1000.0, 0.0, 0.0],   // origin
    [2000.0, 0.0, 0.0],    // strike direction
    [0.0, 0.0, -1000.0],   // down-dip direction
    20,
    10
);

const fault = model.addSurface(positions, indices);

// "free" = traction-free (slip allowed); "fixed" = displacement constrained.
fault.setBcType("dip", "free");     // allow dip-slip
fault.setBcType("strike", "free");  // allow strike-slip
fault.setBcType("normal", "fixed"); // no fault opening/closing

console.log(`Fault: ${fault.nbTriangles()} triangles, ${fault.nbVertices()} vertices`);
console.log(`Model dof: ${model.nbDof()}`);

// ---------------------------------------------------------------------------
// 3. Far-field (remote) stress
// ---------------------------------------------------------------------------
// The background stress state applied at infinity, returned flat as
// [Sxx, Sxy, Sxz, Syy, Syz, Szz] (Pa). A lithostatic-ish state with horizontal
// anisotropy. The fault plane is x-z (normal along y), so its traction is
// sigma . [0,1,0] = [Sxy, Syy, Syz]: a purely diagonal stress would give only a
// normal traction (no slip). The Sxy term is the shear that drives strike-slip.
// UserRemote(fn, true) marshals this JS callback into the C++ std::function and
// evaluates it per point, giving a stress that varies with depth.
function remoteStress(x, y, z) {
    const Sv = -2500.0 * 9.81 * Math.abs(z); // vertical, compressive with depth
    const SHmax = 0.8 * Sv;
    const Shmin = 0.5 * Sv;
    const Sxy = 0.4 * Sv;                     // shear on the fault -> strike-slip
    return [Shmin, Sxy, 0.0, SHmax, 0.0, Sv];
}

const remote = new arch3.UserRemote(remoteStress, true); // true => the field is a stress
model.addRemote(remote);

// ---------------------------------------------------------------------------
// 4. Solve (Gauss-Seidel iterative BEM solver)
// ---------------------------------------------------------------------------
const solver = new arch3.SeidelSolver(model);
solver.setEps(1e-9);
solver.setMaxIter(200);
solver.onMessage((msg) => console.log(`  [solver] ${msg}`));
solver.onEnd(() => console.log("  [solver] done"));

console.log("\n--- Solving ---");
const ok = solver.run(true);
console.log(`Solver converged: ${ok}`);

// ---------------------------------------------------------------------------
// 5. Post-process: slip on the fault, and stress/displacement in the medium
// ---------------------------------------------------------------------------
const post = new arch3.Postprocess(model);

// Burgers vectors (slip) on each triangle, local frame: [b0x,b0y,b0z, ...]
const burgers = post.burgersFor(fault, true, true); // local=true, atTriangles=true
const slipMag = [];
for (let i = 0; i < burgers.length; i += 3) {
    slipMag.push(Math.hypot(burgers[i], burgers[i + 1], burgers[i + 2]));
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
console.log("\n--- Slip on fault ---");
console.log(`  max |slip|  = ${Math.max(...slipMag).toExponential(4)} m`);
console.log(`  mean |slip| = ${mean(slipMag).toExponential(4)} m`);

// Field at observation points OFF the fault plane (y != 0; on-plane is singular).
const obs = [];
for (let k = 0; k < 11; ++k) obs.push(0.0, 200.0, -100.0 * k);

const disp = post.displ(obs);    // flat [ux,uy,uz, ...]
const stress = post.stress(obs); // flat [Sxx,Sxy,Sxz,Syy,Syz,Szz, ...]

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
