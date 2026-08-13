#!/usr/bin/env python3
# ============================================================================
# Forward BEM model in a tabular (horizontally layered) medium —
# arch.TabularMedium
#
# arch models a homogeneous elastic medium. `TabularMedium` adds a first-order
# moduli-perturbation correction for a horizontally layered medium, after
# Du, Segall & Gao (1994, 1997): the medium is the Model material (shear
# modulus mu_ref) plus piecewise-constant shear-modulus jumps across horizontal
# interfaces, each interface at elevation z carrying
#
#       deltaMu = mu(above) - mu(below)
#
# and contributing a surface integral of the homogeneous-reference stress
# against the Mindlin point-force Green's function.
#
# ---------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# Solves the SAME forward model twice — once in the homogeneous half-space,
# once with a TabularMedium attached — and compares, field by field:
#
#   * the Burger's vectors (slip) on the fault,
#   * displacement and strain on a VERTICAL grid (an x-z cross-section beside
#     the fault),
#   * displacement and strain on the TOP SURFACE grid (z = 0).
#
# ---------------------------------------------------------------------------
# TWO THINGS TO KNOW
# ---------------------------------------------------------------------------
# 1. `Model.setTabularMedium()` stores a RAW POINTER. Keep the python object
#    bound to a name that outlives the model, or you read freed memory:
#
#        model.setTabularMedium(arch.TabularMedium())   # WRONG — dangles
#
#        medium = arch.TabularMedium()                  # right
#        model.setTabularMedium(medium)
#
# 2. It is a first-order perturbation: meaningful for |deltaMu| / mu_ref < 1,
#    and a fault must not cross an interface (the interface integrand is
#    singular there). This script asserts the second condition.
# ============================================================================

import math
import os
import sys

# Make the freshly built binding importable regardless of CWD.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "bindings", "python-expanded"
    ),
)

import arch


# ============================================================================
# Model constants
# ============================================================================
POISSON, YOUNG, DENSITY = 0.25, 30e9, 2500.0
MU_REF = YOUNG / (2.0 * (1.0 + POISSON))  # 12 GPa — the reference shear modulus

# Vertical fault striking along x, buried between z = -500 m and z = -3000 m.
FAULT_HALF_LENGTH = 2000.0
FAULT_TOP, FAULT_BOTTOM = -500.0, -3000.0
N_STRIKE, N_DIP = 8, 5

# Interfaces, 1 km BELOW the fault bottom and deeper. They must not cut the
# fault. deltaMu = mu(above) - mu(below): positive means a softer layer below.
INTERFACES = [(-4000.0, 4e9), (-6000.0, -2e9)]
QUADRATURE = 24  # n x n Gauss-Legendre points per interface

# arch local axes, in order: (normal, strike, dip)
AXES = ("normal", "strike", "dip")


# ============================================================================
# Fault mesh: rectangular grid of quads split into triangles
# ============================================================================
def fault_mesh():
    """Returns (positions, indices) as the flat lists arch expects."""
    positions = []
    for j in range(N_DIP + 1):
        for i in range(N_STRIKE + 1):
            x = -FAULT_HALF_LENGTH + 2.0 * FAULT_HALF_LENGTH * i / N_STRIKE
            z = FAULT_TOP + (FAULT_BOTTOM - FAULT_TOP) * j / N_DIP
            positions.extend([x, 0.0, z])

    indices = []
    for j in range(N_DIP):
        for i in range(N_STRIKE):
            v0 = j * (N_STRIKE + 1) + i
            v1, v2 = v0 + 1, v0 + (N_STRIKE + 1)
            indices.extend([v0, v1, v2, v1, v2 + 1, v2])
    return positions, indices


POSITIONS, INDICES = fault_mesh()

for _z, _ in INTERFACES:
    assert not FAULT_BOTTOM <= _z <= FAULT_TOP, f"interface z={_z} cuts the fault"


# ============================================================================
# Observation grids
# ============================================================================
def vertical_grid():
    """An x-z cross-section beside the fault plane.

    It sits at y = 500 m: the BEM kernels are singular ON the fault plane
    (y = 0), so an observation point there yields NaN.
    """
    points = []
    for k in range(13):  # z from 0 down to -6000
        for i in range(9):  # x from -3200 to 3200
            points.extend([-3200.0 + 800.0 * i, 500.0, -500.0 * k])
    return points


def surface_grid():
    """The free surface, z = 0, above and around the fault."""
    points = []
    for j in range(9):
        for i in range(9):
            points.extend([-3200.0 + 800.0 * i, -3200.0 + 800.0 * j, 0.0])
    return points


VERTICAL = vertical_grid()
SURFACE = surface_grid()


# ============================================================================
# The forward model
# ============================================================================
def remote_stress(x, y, z):
    """Lithostatic-ish background with a shear term driving slip.

    Returned flat as [Sxx, Sxy, Sxz, Syy, Syz, Szz] in Pa. The fault plane is
    x-z (normal along y), so its traction is sigma . [0,1,0] = [Sxy, Syy, Syz]:
    the Sxy term is the shear that drives strike slip.
    """
    Sv = -DENSITY * 9.81 * abs(z)
    return [0.5 * Sv, 0.4 * Sv, 0.0, 0.8 * Sv, 0.0, Sv]


def run(medium=None):
    """Solve the forward model and post-process. `medium` is a TabularMedium
    or None for the homogeneous half-space."""
    model = arch.Model()
    model.setHalfSpace(True)
    model.setMaterial(POISSON, YOUNG, DENSITY)

    fault = model.addSurface(POSITIONS, INDICES)
    fault.setBcType("strike", "free")  # slip allowed along strike
    fault.setBcType("dip", "free")  # and down dip
    fault.setBcType("normal", "fixed")  # no opening / interpenetration

    remote = arch.UserRemote(remote_stress, True)  # True => the field is a stress
    model.addRemote(remote)

    if medium is not None:
        model.setTabularMedium(medium)

    solver = arch.SeidelSolver(model)
    solver.setEps(1e-10)
    solver.setMaxIter(500)
    converged = solver.run(False)

    post = arch.Postprocess(model)
    return {
        "converged": converged,
        "burgers": post.burgersFor(fault, True, True),  # local frame, at triangles
        "displ_vertical": post.displ(VERTICAL),
        "strain_vertical": post.strain(VERTICAL),
        "displ_surface": post.displ(SURFACE),
        "strain_surface": post.strain(SURFACE),
    }


# ============================================================================
# Comparison helpers
# ============================================================================
def compare(reference, other):
    """Max absolute difference, the same relative to the reference amplitude,
    and the RMS difference."""
    worst = max((abs(a - b) for a, b in zip(reference, other)), default=0.0)
    scale = max((abs(a) for a in reference), default=0.0)
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(reference, other)) / len(reference))
    return worst, (worst / scale if scale else float("nan")), rms


def report(title, fields, homogeneous, layered):
    print(f"\n--- {title} ---")
    print(f"  {'field':<18} {'amplitude':>12} {'max |diff|':>12} {'rel':>9} {'RMS diff':>12}")
    for key, label in fields:
        worst, rel, rms = compare(homogeneous[key], layered[key])
        scale = max(abs(a) for a in homogeneous[key])
        print(f"  {label:<18} {scale:12.4e} {worst:12.4e} {rel:8.3%} {rms:12.4e}")


# ============================================================================
# Run
# ============================================================================
def main():
    print(f"Fault      : {len(INDICES) // 3} triangles, z in [{FAULT_BOTTOM}, {FAULT_TOP}] m")
    print(f"mu_ref     : {MU_REF / 1e9:.1f} GPa")
    print("Interfaces :")
    for z, dmu in INTERFACES:
        print(f"    z = {z:8.1f} m   deltaMu = {dmu / 1e9:+5.1f} GPa   "
              f"|dMu|/mu_ref = {abs(dmu) / MU_REF:.3f}")
    print(f"Grids      : vertical {len(VERTICAL) // 3} pts (x-z at y=500), "
          f"surface {len(SURFACE) // 3} pts (z=0)")

    print("\n--- Solving in the homogeneous half-space ---")
    homogeneous = run()
    print(f"  converged: {homogeneous['converged']}")

    # Bound to a name: Model keeps only a raw pointer to it (see note 1).
    medium = arch.TabularMedium()
    for z, dmu in INTERFACES:
        medium.addInterface(z, dmu)
    medium.setQuadrature(QUADRATURE)

    print(f"\n--- Solving with the tabular medium ({medium.nbInterfaces()} interfaces) ---")
    layered = run(medium)
    print(f"  converged: {layered['converged']}")

    # --- slip on the fault, per local axis --------------------------------
    print("\n--- Burger's vectors on the fault (local frame, m) ---")
    print(f"  {'axis':<10} {'homogeneous':>14} {'layered':>14} {'max |diff|':>14}")
    for a, axis in enumerate(AXES):
        bh = homogeneous["burgers"][a::3]
        bl = layered["burgers"][a::3]
        worst = max(abs(x - y) for x, y in zip(bh, bl))
        print(f"  {axis:<10} {max(abs(v) for v in bh):14.6e} "
              f"{max(abs(v) for v in bl):14.6e} {worst:14.6e}")

    # --- fields on the two grids ------------------------------------------
    report(
        "Vertical grid (x-z cross-section at y = 500 m)",
        [("displ_vertical", "displacement"), ("strain_vertical", "strain")],
        homogeneous,
        layered,
    )
    report(
        "Top surface grid (z = 0)",
        [("displ_surface", "displacement"), ("strain_surface", "strain")],
        homogeneous,
        layered,
    )

    # --- verdict ----------------------------------------------------------
    keys = ("burgers", "displ_vertical", "strain_vertical", "displ_surface", "strain_surface")
    changed = [k for k in keys if compare(homogeneous[k], layered[k])[0] > 0.0]
    print("\n--- Verdict ---")
    if changed:
        print("  the tabular medium changed: " + ", ".join(changed))
    else:
        print(
            "  Every field is bit-for-bit identical: the forward path does not\n"
            "  see the tabular medium. TabularMedium::correction is applied only\n"
            "  in DisplCoefficients::influence (DisplCoefficients.cxx:56), and\n"
            "  the forward path never calls it:\n"
            "    - the solver assembles tractions through StrainCoefficients,\n"
            "    - Postprocess.displ  -> DisplCoefficients::value  (no correction),\n"
            "    - Postprocess.strain -> StrainCoefficients        (no correction).\n"
            "  Only influence-matrix consumers — the slip inversion — inherit the\n"
            "  layered medium today. Wiring the correction into ::value and into\n"
            "  StrainCoefficients is what this script would measure."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
