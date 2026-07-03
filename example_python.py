#!/usr/bin/env python3
# ============================================================================
# Forward Boundary Element Method (BEM) model using the rosetta-generated
# `arch3` python binding.
#
# This is the rosetta equivalent of scripts/forward.py — the simplest
# meaningful arch3 workflow:
#
#       Model -> Surface (fault) -> boundary conditions -> remote stress
#             -> iterative solve -> post-process displacement & stress
#
# Unlike scripts/forward.py it is fully self-contained: the fault mesh is
# generated in-code (no .ts file, no xali_tools dependency), so it runs
# against nothing but the freshly built `arch3` module in ./bindings/python.
#
# NOTE on API differences vs. the hand-written `pyarch3` used in scripts/:
#   - Surfaces are created with `model.addSurface(positions, indices)`
#     (returns a Surface), not `arch3.Surface(model, ...)`.
#   - `UserRemote(fn, isStress=True)` takes the stress/strain flag. It is passed
#     to `Model.addRemote()`, which expects a `BaseRemote` — the rosetta python
#     binding registers the `UserRemote : BaseRemote` inheritance, so this works.
# ============================================================================

import os
import sys

# Make the freshly built binding importable regardless of CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bindings", "python"))

import arch3


# ============================================================================
# Helper: build a flat rectangular triangulated patch (the fault surface)
# ============================================================================
def make_strip_mesh(origin, axis1, axis2, n1, n2):
    """Rectangular grid of n1 x n2 quads -> 2*n1*n2 triangles.

    Returns (positions, indices) as the flat lists arch3 expects:
      positions = [x0,y0,z0, x1,y1,z1, ...]
      indices   = [i0,i1,i2, i3,i4,i5, ...]
    """
    positions = []
    for j in range(n2 + 1):
        for i in range(n1 + 1):
            positions.extend(
                origin[k] + axis1[k] * i / n1 + axis2[k] * j / n2 for k in range(3)
            )

    indices = []
    for j in range(n2):
        for i in range(n1):
            v0 = j * (n1 + 1) + i
            v1 = v0 + 1
            v2 = v0 + (n1 + 1)
            v3 = v2 + 1
            indices.extend([v0, v1, v2, v1, v3, v2])

    return positions, indices


# ============================================================================
# 1. Model + material
# ============================================================================
model = arch3.Model()
model.setHalfSpace(False)
# setMaterial(poisson, young [Pa], density)
model.setMaterial(0.25, 20e9, 2500)


# ============================================================================
# 2. Fault surface (the discontinuity)
# ============================================================================
# A vertical fault, 2 km along strike (x) x 1 km down dip (-z), centred at y=0.
positions, indices = make_strip_mesh(
    origin=[-1000.0, 0.0, 0.0],
    axis1=[2000.0, 0.0, 0.0],   # strike direction
    axis2=[0.0, 0.0, -1000.0],  # down-dip direction
    n1=20,
    n2=10,
)

fault = model.addSurface(positions, indices)

# Boundary conditions per local component:
#   "free"  = traction-free (slip allowed in that direction)
#   "fixed" = displacement constrained (no opening/closing)
fault.setBcType("dip", "free")     # allow dip-slip
fault.setBcType("strike", "free")  # allow strike-slip
fault.setBcType("normal", "fixed") # no fault opening/closing

print(f"Fault: {fault.nbTriangles()} triangles, {fault.nbVertices()} vertices")
print(f"Model dof: {model.nbDof()}")


# ============================================================================
# 3. Far-field (remote) stress
# ============================================================================
# The background stress state applied at infinity, returned flat as
# [Sxx, Sxy, Sxz, Syy, Syz, Szz] (Pa). A lithostatic-ish state with horizontal
# anisotropy. The fault plane is x-z (normal along y), so its traction is
# sigma . [0,1,0] = [Sxy, Syy, Syz]: a purely diagonal stress would give only a
# normal traction (no slip). The Sxy term is the shear that drives strike-slip —
# equivalent to loading an obliquely-oriented fault with principal stresses.
def remote_stress(x, y, z):
    Sv = -2500.0 * 9.81 * abs(z)  # vertical, compressive (negative) with depth
    SHmax = 0.8 * Sv
    Shmin = 0.5 * Sv
    Sxy = 0.4 * Sv                # shear on the fault → strike-slip
    return [Shmin, Sxy, 0.0, SHmax, 0.0, Sv]


remote = arch3.UserRemote(remote_stress, True)  # True => the field is a stress
model.addRemote(remote)


# ============================================================================
# 4. Solve (Gauss-Seidel iterative BEM solver)
# ============================================================================
solver = arch3.SeidelSolver(model)
solver.setEps(1e-9)
solver.setMaxIter(200)
solver.onMessage(lambda msg: print(f"  [solver] {msg}"))
solver.onEnd(lambda: print("  [solver] done"))

print("\n--- Solving ---")
ok = solver.run(True)
print(f"Solver converged: {ok}")


# ============================================================================
# 5. Post-process: slip on the fault, and stress/displacement in the medium
# ============================================================================
post = arch3.Postprocess(model)

# Burgers vectors (slip) on each triangle, in the local fault frame.
burgers = post.burgersFor(fault, True, True)  # local=True, atTriangles=True
slip_mag = [
    (burgers[i] ** 2 + burgers[i + 1] ** 2 + burgers[i + 2] ** 2) ** 0.5
    for i in range(0, len(burgers), 3)
]
print("\n--- Slip on fault ---")
print(f"  max |slip|  = {max(slip_mag):.4e} m")
print(f"  mean |slip| = {sum(slip_mag) / len(slip_mag):.4e} m")

# Evaluate the perturbed field at a line of observation points beside the
# fault. They must sit OFF the fault plane (y != 0), where the BEM kernels are
# regular — a point on the surface itself is singular and yields NaN.
obs = []
for k in range(11):
    obs.extend([0.0, 200.0, -100.0 * k])  # 11 points going down, 200 m off-fault

disp = post.displ(obs)       # flat [ux,uy,uz, ...]
stress = post.stress(obs)    # flat [Sxx,Sxy,Sxz,Syy,Syz,Szz, ...]

print("\n--- Field at observation points (x=0, y=200) ---")
print(f"  {'z (m)':>8} {'|u| (m)':>12} {'Szz (Pa)':>14}")
for k in range(len(obs) // 3):
    z = obs[k * 3 + 2]
    ux, uy, uz = disp[k * 3 : k * 3 + 3]
    umag = (ux * ux + uy * uy + uz * uz) ** 0.5
    szz = stress[k * 6 + 5]
    print(f"  {z:8.0f} {umag:12.4e} {szz:14.4e}")

print("\nDone.")
