#!/usr/bin/env python3
# =============================================================================
#  Linearized Coulomb evaluation by superposition — worked example
#
#  Demonstrates `arch::CoulombSuperposition` (see arch3/docs/coulomb-superposition.md)
#  on a real two-fault geological model read from a Gocad TSurf file.
#
#  Pipeline
#  --------
#    1. read the faults (FaultS1 + FaultS2) from s1.ts
#    2. build N = 5 linearly independent BEM basis solutions
#         -> per-element driving tractions (Sn, Ss, Sd), local frame
#         -> observation stress (Sxx..Szz), global frame
#    3. verify that superposition is EXACT (vs. a direct re-solve)
#    4. evaluate the Coulomb criterion for a candidate stress state
#    5. map the frictional-admissibility penalty PI over (theta, R)
#    6. run a Monte Carlo stress inversion on synthetic fracture data,
#       with and without the Coulomb penalty, and compare
#
#  Run:  python3 coulomb_superposition_s1.py [/path/to/s1.ts]
# =============================================================================

import glob
import importlib.util
import math
import os
import sys
import time

import numpy as np

TS_FILE = sys.argv[1] if len(sys.argv) > 1 else "/Users/fmaerten/devs/data/s1.ts"

# -----------------------------------------------------------------------------
# Load the freshly built rosetta binding by file path (bypasses any stale
# site-packages install put on sys.meta_path by an editable install).
# -----------------------------------------------------------------------------
_BINDING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "nanobind"
)
_candidates = sorted(glob.glob(os.path.join(_BINDING_DIR, "arch*.so")))
if not _candidates:
    sys.exit(f"No built `arch` extension found in {_BINDING_DIR} — build it first.")
_spec = importlib.util.spec_from_file_location("arch", _candidates[0])
arch = importlib.util.module_from_spec(_spec)
sys.modules["arch"] = arch
_spec.loader.exec_module(arch)


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# =============================================================================
# 0. Gocad TSurf reader
# =============================================================================
def read_tsurf(path):
    """Minimal Gocad TSurf reader.

    A .ts file may hold several TSurf objects; vertex ids restart at 1 in each.
    Returns [(name, positions_flat, indices_flat_0based), ...].
    """
    surfaces, name, vmap, pos, idx = [], None, {}, [], []

    def flush():
        if pos and idx:
            surfaces.append((name or f"surface{len(surfaces)}", pos[:], idx[:]))

    with open(path) as fh:
        for line in fh:
            f = line.split()
            if not f:
                continue
            key = f[0]
            if key == "GOCAD":
                flush()
                name, vmap, pos, idx = None, {}, [], []
            elif key == "name:":
                name = f[1]
            elif key in ("VRTX", "PVRTX"):
                vmap[int(f[1])] = len(vmap)
                pos.extend((float(f[2]), float(f[3]), float(f[4])))
            elif key == "ATOM":  # shared vertex: alias onto the referenced one
                vmap[int(f[1])] = vmap[int(f[2])]
            elif key == "TRGL":
                idx.extend(vmap[int(f[i])] for i in (1, 2, 3))
            elif key == "END":
                flush()
                name, vmap, pos, idx = None, {}, [], []
    flush()
    return surfaces


# =============================================================================
# 1. Model
# =============================================================================
rule("1. MODEL")

POISSON, YOUNG, DENSITY = 0.25, 30e9, 2700.0
G = 9.81

model = arch.Model()
model.setHalfSpace(True)  # free surface at z = 0
model.setMaterial(POISSON, YOUNG, DENSITY)

tsurfs = read_tsurf(TS_FILE)
faults, fault_names = [], []
for nm, pos, idx in tsurfs:
    s = model.addSurface(pos, idx)
    # Frictionless basis: the fault is free to slip in its plane, no opening.
    s.setBcType("dip", "free")
    s.setBcType("strike", "free")
    s.setBcType("normal", "fixed")
    faults.append(s)
    fault_names.append(nm)
    print(f"  {nm:10s}  {s.nbTriangles():4d} triangles  {s.nbVertices():4d} vertices")

n_elts = sum(s.nbTriangles() for s in faults)
print(f"  total     {n_elts:4d} elements, model dof = {model.nbDof()}")

bounds = model.bounds()  # [xmin, ymin, zmin, xmax, ymax, zmax]
print(
    f"  bounds    x[{bounds[0]:.0f}, {bounds[3]:.0f}]  "
    f"y[{bounds[1]:.0f}, {bounds[4]:.0f}]  z[{bounds[2]:.0f}, {bounds[5]:.0f}] m"
)

# Element centres (used to place observation points away from the faults).
centres = np.array([t.center().asArray() for t in model.triangles()])

# -----------------------------------------------------------------------------
# Observation points: a horizontal grid at mid-fault depth, keeping only the
# points far enough from every element (the BEM kernels are singular ON the
# discontinuity).
# -----------------------------------------------------------------------------
Z_OBS = 0.5 * (bounds[2] + bounds[5])
MIN_DIST = 400.0
obs_pts = []
for x in np.linspace(bounds[0], bounds[3], 11):
    for y in np.linspace(bounds[1], bounds[4], 11):
        p = np.array([x, y, Z_OBS])
        if np.min(np.linalg.norm(centres - p, axis=1)) > MIN_DIST:
            obs_pts.append(p)
obs_pts = np.array(obs_pts)
obs_flat = obs_pts.ravel().tolist()
n_obs = len(obs_pts)
print(f"  observation points: {n_obs} at z = {Z_OBS:.0f} m (>{MIN_DIST:.0f} m off-fault)")


# =============================================================================
# 2. Basis construction
# =============================================================================
rule("2. BASIS CONSTRUCTION  (N linearly independent BEM solves)")

# Each basis mode is a unit far-field stress state. The `arch` convention is
# COMPRESSION NEGATIVE for the stress tensor, returned flat as
#   [Sxx, Sxy, Sxz, Syy, Syz, Szz]
BASIS = [
    ("Sxx", lambda x, y, z: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("Sxy", lambda x, y, z: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    ("Syy", lambda x, y, z: [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
    ("Szz", lambda x, y, z: [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    # Lithostatic gradient: rho.g.z (z < 0 => compressive). This mode carries
    # the absolute stress level and is what makes the friction problem
    # scale-dependent (see step 5).
    (
        "litho",
        lambda x, y, z: [
            DENSITY * G * z,
            0.0,
            0.0,
            DENSITY * G * z,
            0.0,
            DENSITY * G * z,
        ],
    ),
]
N = len(BASIS)

cs = arch.CoulombSuperposition()
basis_tractions, basis_obs = [], []

t0 = time.time()
for k, (label, fn) in enumerate(BASIS):
    model.clearRemotes()
    remote = arch.UserRemote(fn, True)  # True => the field is a stress
    model.addRemote(remote)

    # --- (a) the mechanical solve --------------------------------------------
    # The basis is FRICTIONLESS: the faults slip freely in their plane. This
    # keeps the boundary conditions independent of the loading, which is the
    # condition for superposition to hold exactly.
    solver = arch.SeidelSolver(model)
    solver.setEps(1e-9)
    solver.setMaxIter(500)
    solver.run(True)

    # --- (b) driving traction on every fault element, LOCAL frame ------------
    # Model.tractions() returns one Serie(3, nTri) per surface holding the
    # remote traction resolved on each element (normal, strike, dip), in the
    # BOUNDARY-CONDITION convention (compression NEGATIVE).
    #
    # CoulombSuperposition expects the SOLVER/RHS convention, in which
    # Sn > 0 means compression (System::update_b negates the BC values).
    # Hence the sign flip below — getting this wrong turns every compressive
    # element into a tensile one and the criterion becomes meaningless.
    #
    # NOTE: this must come AFTER run(), because the local frames are built
    # lazily when the influence coefficients are first assembled.
    tract = []
    for serie in model.tractions():
        tract.extend(-v for v in serie.asArray())
    t_serie = arch.Serie(3, tract)

    # --- (c) observation stress, GLOBAL frame --------------------------------
    # Far field plus the perturbation radiated by the slipping faults.
    post = arch.Postprocess(model)
    s_serie = arch.Serie(6, post.stress(obs_flat))

    assert cs.addBasis(t_serie, s_serie)
    basis_tractions.append(np.array(tract).reshape(-1, 3))
    basis_obs.append(np.array(post.stress(obs_flat)).reshape(-1, 6))
    print(f"  [{k}] {label:6s} basis solved")

t_basis = time.time() - t0
print(f"\n  {N} basis solutions in {t_basis:.2f} s")
print(
    f"  stored: {cs.nbBasis()} bases x {cs.nbFaultElements()} elements "
    f"x {cs.nbObsPoints()} observation points"
)


# =============================================================================
# 3. Superposition is exact — verification against a direct re-solve
# =============================================================================
rule("3. EXACTNESS CHECK  (superposition vs. direct BEM re-solve)")

alpha_chk = [12e6, -4e6, 6e6, 3e6, 1.0]

superposed = np.array(cs.superposeObsStress(alpha_chk).asArray()).reshape(-1, 6)


def combined_remote(x, y, z):
    """The single far-field stress equal to sum_k alpha_k * sigma_(k)."""
    out = [0.0] * 6
    for a, (_, fn) in zip(alpha_chk, BASIS):
        for i, v in enumerate(fn(x, y, z)):
            out[i] += a * v
    return out


# Superposition has no error of its own, so the discrepancy against a direct
# re-solve should be governed entirely by how well the solves converged.
# Sweeping the DIRECT solve's tolerance shows exactly that: the mismatch tracks
# eps down to the floor inherited from the basis solves (themselves run at 1e-9).
print(f"  {'direct eps':>12s} {'rel. mismatch':>15s} {'solve time':>12s}")
t_solve = None
for eps in (1e-4, 1e-6, 1e-9, 1e-12):
    model.clearRemotes()
    remote = arch.UserRemote(combined_remote, True)
    model.addRemote(remote)
    solver = arch.SeidelSolver(model)
    solver.setEps(eps)
    solver.setMaxIter(2000)
    t0 = time.time()
    solver.run(True)
    dt = time.time() - t0
    direct = np.array(arch.Postprocess(model).stress(obs_flat)).reshape(-1, 6)
    err = np.abs(superposed - direct).max() / np.abs(direct).max()
    print(f"  {eps:12.0e} {err:15.3e} {dt*1000:9.1f} ms")
    if eps == 1e-9:
        t_solve = dt

t0 = time.time()
for _ in range(1000):
    cs.superposeObsStress(alpha_chk)
t_sup = (time.time() - t0) / 1000
print(f"\n  superposition    : {t_sup*1000:8.4f} ms   -> speed-up "
      f"x{t_solve/t_sup:,.0f} over the eps=1e-9 solve")
print("\n  => the mismatch is the iterative solver's own convergence error, not a")
print("     superposition error: it shrinks with eps and floors at the accuracy")
print("     of the basis. Superposition of the BEM response is EXACT, because the")
print("     influence matrix does not depend on the loading.")


# =============================================================================
# 4. Coulomb criterion for one candidate stress state
# =============================================================================
rule("4. COULOMB EVALUATION AT ONE CANDIDATE STATE")

MU, COHESION, LAMBDA = 0.6, 0.0, 0.6
cs.setFriction(MU)
cs.setCohesion(COHESION)
cs.setLambda(LAMBDA)
print(f"  friction mu = {MU},  cohesion c = {COHESION/1e6:.1f} MPa, "
      f"fluid-pressure ratio lambda = {LAMBDA}")
print(f"  => effective strength  tau_max = mu (1-lambda) Sn "
      f"= {MU*(1-LAMBDA):.2f} Sn")


def anderson(theta_deg, R):
    """Andersonian stress tensor -> [Sxx, Sxy, Syy, Szz] (compression negative).

    Principal values normalized to  s1 = 0 >= s2 = R-1 >= s3 = -1, one principal
    axis vertical.  theta = azimuth of SHmax from +x.
        R in [0,1] : normal faulting     (sigma_v most compressive)
        R in [1,2] : strike-slip         (sigma_v intermediate)
        R in [2,3] : reverse faulting    (sigma_v least compressive)
    Mirrors arch::sinv::mapping::fullAnderson.
    """
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    c2, s2 = c * c, s * s
    if R <= 1.0:  # normal faulting: sigma_v most compressive
        xx, xy, yy, zz = R * s2, R * c * s, R * c2, 1.0
    elif R <= 2.0:  # strike-slip: sigma_v intermediate
        r = 2.0 - R
        xx, xy, yy, zz = s2, c * s, c2, r
    else:  # reverse faulting: sigma_v least compressive
        r = R - 2.0
        xx, xy, yy, zz = r * c2 + s2, (1.0 - r) * c * s, r * s2 + c2, 0.0
    return -xx, -xy, -yy, -zz  # negate: compression is negative


def alpha_of(theta_deg, R, magnitude):
    """(theta, R, magnitude) -> superposition coefficients.

    The lithostatic mode always has coefficient 1: the deviatoric tectonic
    stress of amplitude `magnitude` (Pa) rides on top of it.
    """
    xx, xy, yy, zz = anderson(theta_deg, R)
    return [magnitude * xx, magnitude * xy, magnitude * yy, magnitude * zz, 1.0]


THETA0, R0, MAG0 = 35.0, 1.5, 50e6  # strike-slip regime, SHmax at N35E-ish
alpha0 = alpha_of(THETA0, R0, MAG0)
print(f"  candidate: theta = {THETA0}deg, R = {R0} (strike-slip), "
      f"deviatoric magnitude = {MAG0/1e6:.0f} MPa")

ev = np.array(cs.evaluateCoulomb(alpha0).asArray()).reshape(-1, 4)
excess, sliding, dir_s, dir_d = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
tr = np.array(cs.superposeFaultTractions(alpha0).asArray()).reshape(-1, 3)
Sn, Ss, Sd = tr[:, 0], tr[:, 1], tr[:, 2]
tau = np.hypot(Ss, Sd)

print(f"\n  {'':12s} {'elements':>9s} {'sliding':>9s} {'max excess':>13s} "
      f"{'mean |tau|':>12s} {'mean Sn':>10s}")
off = 0
for name, s in zip(fault_names, faults):
    n = s.nbTriangles()
    sl = slice(off, off + n)
    print(
        f"  {name:12s} {n:9d} {int(sliding[sl].sum()):9d} "
        f"{excess[sl].max()/1e6:10.2f} MPa {tau[sl].mean()/1e6:9.2f} MPa "
        f"{Sn[sl].mean()/1e6:7.2f} MPa"
    )
    off += n
print(
    f"  {'ALL':12s} {n_elts:9d} {cs.nbSliding(alpha0):9d} "
    f"{excess.max()/1e6:10.2f} MPa {tau.mean()/1e6:9.2f} MPa {Sn.mean()/1e6:7.2f} MPa"
)
print(f"\n  Coulomb penalty  PI = {cs.coulombPenalty(alpha0):.4f}   "
      f"(0 = fully admissible)")
print(f"  fraction of the fault system at/over failure: "
      f"{100*sliding.mean():.1f} %")


def rake_stats(ds, dd):
    """Statistics of a set of slip directions (dir_s, dir_d).

    Rakes are angles on a circle, so the arithmetic mean is meaningless. Worse,
    a fault system usually slips in BOTH senses (right- and left-lateral on
    differently oriented patches), giving an antipodal, bimodal distribution
    whose vector mean cancels. The slip AXIS is what is well defined, so use
    axial (double-angle) statistics.

    Returns (mean_axis_deg in [-90, 90), axial_spread_deg, vector_resultant).
    """
    if len(ds) == 0:
        return float("nan"), float("nan"), 0.0
    ang = np.arctan2(dd, ds)
    c2, s2 = np.cos(2 * ang).mean(), np.sin(2 * ang).mean()
    Rax = math.hypot(c2, s2)                      # axial concentration
    mean = math.degrees(0.5 * math.atan2(s2, c2))  # mean slip axis
    spread = (0.5 * math.degrees(math.sqrt(-2.0 * math.log(Rax)))
              if Rax > 1e-12 else float("inf"))
    Rvec = math.hypot(ds.mean(), dd.mean())       # vector (signed) resultant
    return mean, spread, Rvec


# Predicted slip direction on the elements that do slip. The rake is measured
# from the strike direction: 0 = pure strike-slip, +/-90 = pure dip-slip.
if sliding.sum():
    axis, spread, Rvec = rake_stats(dir_s[sliding > 0], dir_d[sliding > 0])
    rk = np.degrees(np.arctan2(dir_d[sliding > 0], dir_s[sliding > 0]))
    frac_ss = float((np.abs(np.abs(rk) - 90.0) > 45.0).mean())
    print(f"  predicted slip axis on sliding elements: {axis:6.1f} deg from strike "
          f"(+/- {spread:.0f} deg)")
    print(f"    {100*frac_ss:.0f} % strike-slip-dominated, "
          f"{100*(1-frac_ss):.0f} % dip-slip-dominated")
    if Rvec < 0.3:
        print(f"    vector resultant only {Rvec:.2f}: the two faults slip in "
              f"OPPOSITE senses (bimodal), hence the axial statistic")

print("\n  Most critical elements (largest Coulomb excess):")
print(f"    {'#':>5s} {'Sn (MPa)':>10s} {'tau (MPa)':>10s} {'tau_max':>10s} "
      f"{'excess':>9s} {'rake':>7s}")
tau_max = np.where(Sn > 0, MU * (1 - LAMBDA) * Sn + COHESION, COHESION)
for i in np.argsort(-excess)[:6]:
    print(
        f"    {i:5d} {Sn[i]/1e6:10.2f} {tau[i]/1e6:10.2f} {tau_max[i]/1e6:10.2f} "
        f"{excess[i]/1e6:9.2f} {math.degrees(math.atan2(dir_d[i], dir_s[i])):7.1f}"
    )


# =============================================================================
# 5. Frictional-admissibility map over (theta, R)
# =============================================================================
rule("5. ADMISSIBILITY MAP  PI(theta, R)   [magnitude fixed]")

# The stress state that section 6 will try to recover from data. Declared here so
# the map can show where it sits.
THETA_TRUE, R_TRUE, MAG_TRUE = 62.0, 1.35, 30e6

MAG_MAP = MAG_TRUE
print(f"  deviatoric magnitude fixed at {MAG_MAP/1e6:.0f} MPa; '+' marks the stress")
print(f"  state used as ground truth in section 6 (theta = {THETA_TRUE:.0f} deg, "
      f"R = {R_TRUE}).")

thetas = np.arange(0.0, 180.0, 10.0)
Rs = np.arange(0.1, 3.0, 0.2)
PI = np.zeros((len(Rs), len(thetas)))

t0 = time.time()
for i, R in enumerate(Rs):
    for j, th in enumerate(thetas):
        PI[i, j] = cs.coulombPenalty(alpha_of(th, R, MAG_MAP))
t_map = time.time() - t0
print(f"  {PI.size} evaluations in {t_map:.2f} s "
      f"({1e6*t_map/PI.size:.0f} us each) — a direct re-solve would take "
      f"{PI.size*t_solve/60:.0f} min")

LEVELS = " .:-=+*#%@"
vmax = PI.max() if PI.max() > 0 else 1.0
print(f"\n  PI, ' ' = admissible ... '@' = PI_max = {vmax:.2f}  (sqrt colour scale)")
ticks = "".join("|" if int(t) % 30 == 0 else " " for t in thetas)
i_true = int(np.argmin(np.abs(Rs - R_TRUE)))
j_true = int(np.argmin(np.abs(thetas - THETA_TRUE)))
for i, R in enumerate(Rs):
    row = [
        LEVELS[min(int(9 * math.sqrt(PI[i, j] / vmax)), 9)] for j in range(len(thetas))
    ]
    if i == i_true:
        row[j_true] = "+"
    regime = "normal" if R <= 1 else ("strike-slip" if R <= 2 else "reverse")
    print(f"  R={R:4.1f} {regime:>11s}  |{''.join(row)}|")
print(f"  {'':17s}  |{ticks}|")
print(f"  {'':17s}   theta = 0 .. {int(thetas[-1])} deg, step {int(thetas[1]-thetas[0])}"
      f"  ('|' every 30 deg)")

imin = np.unravel_index(np.argmin(PI), PI.shape)
imax = np.unravel_index(np.argmax(PI), PI.shape)
n_adm = int((PI <= 1e-12).sum())
print(f"\n  most stable   : theta = {thetas[imin[1]]:.0f} deg, R = {Rs[imin[0]]:.1f}"
      f"   ->  PI = {PI[imin]:.4f}, "
      f"{cs.nbSliding(alpha_of(thetas[imin[1]], Rs[imin[0]], MAG_MAP))}/{n_elts} sliding")
print(f"  most unstable : theta = {thetas[imax[1]]:.0f} deg, R = {Rs[imax[0]]:.1f}"
      f"   ->  PI = {PI[imax]:.4f}, "
      f"{cs.nbSliding(alpha_of(thetas[imax[1]], Rs[imax[0]], MAG_MAP))}/{n_elts} sliding")
print(f"  fully admissible orientations at this magnitude: {n_adm}/{PI.size}")
print("\n  This map is the mechanical prior the Coulomb criterion adds to a stress")
print("  inversion: the dark regions are stress states this fault system could")
print("  not sustain without slipping.")


# =============================================================================
# 6. Monte Carlo stress inversion on synthetic fracture data
# =============================================================================
rule("6. MONTE CARLO STRESS INVERSION")

# --- 6a. synthetic "observed" data -------------------------------------------
# Ground truth (declared in section 5). Joints (mode-I fractures) open
# perpendicular to the least compressive principal stress, so their pole is
# parallel to sigma3 — which in the compression-negative convention is the
# LARGEST eigenvalue.
NOISE_DEG = 5.0


def sigma3_dirs(alpha):
    """Poles of the joints predicted by the superposed observation stress."""
    s = np.array(cs.superposeObsStress(alpha).asArray()).reshape(-1, 6)
    m = np.empty((len(s), 3, 3))
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2] = s[:, 0], s[:, 1], s[:, 2]
    m[:, 1, 0], m[:, 1, 1], m[:, 1, 2] = s[:, 1], s[:, 3], s[:, 4]
    m[:, 2, 0], m[:, 2, 1], m[:, 2, 2] = s[:, 2], s[:, 4], s[:, 5]
    w, v = np.linalg.eigh(m)          # ascending eigenvalues
    return v[:, :, 2]                  # least compressive direction


rng = np.random.default_rng(42)
alpha_true = alpha_of(THETA_TRUE, R_TRUE, MAG_TRUE)
poles = sigma3_dirs(alpha_true)
poles = poles + np.radians(NOISE_DEG) * rng.standard_normal(poles.shape)
observed = poles / np.linalg.norm(poles, axis=1, keepdims=True)

print(f"  truth        : theta = {THETA_TRUE} deg, R = {R_TRUE}, "
      f"magnitude = {MAG_TRUE/1e6:.0f} MPa")
print(f"  data         : {n_obs} joint poles, {NOISE_DEG} deg of noise")
print(f"  truth check  : PI(truth) = {cs.coulombPenalty(alpha_true):.4f}, "
      f"{cs.nbSliding(alpha_true)}/{n_elts} elements sliding")


def misfit(alpha):
    """Mean angular deviation (deg) between predicted and observed joint poles."""
    d = np.abs(np.sum(sigma3_dirs(alpha) * observed, axis=1))
    return float(np.degrees(np.arccos(np.clip(d, 0.0, 1.0))).mean())


# --- 6b. the search ----------------------------------------------------------
N_SAMPLES = 4000
W_COULOMB = 30.0  # deg per unit of PI — puts the penalty on the misfit scale

samples = np.column_stack(
    [
        rng.uniform(0.0, 180.0, N_SAMPLES),   # theta
        rng.uniform(0.05, 2.95, N_SAMPLES),   # R
        rng.uniform(2e6, 120e6, N_SAMPLES),   # deviatoric magnitude
    ]
)

t0 = time.time()
records = []
for th, R, mag in samples:
    a = alpha_of(th, R, mag)
    records.append((misfit(a), cs.coulombPenalty(a), th, R, mag))
t_mc = time.time() - t0
records = np.array(records)
print(f"\n  {N_SAMPLES} samples in {t_mc:.1f} s "
      f"({1e3*t_mc/N_SAMPLES:.2f} ms/sample, misfit + penalty)")

D, P, TH, RR, MG = records.T

J_data = D                       # orientation data only
J_full = D + W_COULOMB * P       # data + Coulomb admissibility

best_data = int(np.argmin(J_data))
best_full = int(np.argmin(J_full))

print(f"\n  Best single sample under each cost function:")
print(f"  {'':26s} {'theta':>8s} {'R':>7s} {'magnitude':>12s} "
      f"{'misfit':>9s} {'PI':>8s} {'sliding':>9s}")
print(f"  {'ground truth':26s} {THETA_TRUE:8.1f} {R_TRUE:7.2f} "
      f"{MAG_TRUE/1e6:9.1f} MPa {misfit(alpha_true):7.2f}d "
      f"{cs.coulombPenalty(alpha_true):8.4f} {cs.nbSliding(alpha_true):9d}")
for label, b in (("data misfit only", best_data), ("data + Coulomb penalty", best_full)):
    a = alpha_of(TH[b], RR[b], MG[b])
    print(
        f"  {label:26s} {TH[b]:8.1f} {RR[b]:7.2f} {MG[b]/1e6:9.1f} MPa "
        f"{D[b]:7.2f}d {P[b]:8.4f} {cs.nbSliding(a):9d}"
    )
if best_data == best_full:
    print("  (the two coincide here: the best-fitting sample happens to be")
    print("   frictionally admissible — a single best sample is a poor summary,")
    print("   so compare the posteriors below instead.)")

print("\n  Why the magnitude needs the Coulomb term")
print("  ---------------------------------------")
print("  Joint orientations are scale-free: multiplying the deviatoric stress by")
print("  any kappa > 0 leaves the principal DIRECTIONS unchanged, so the data")
print("  alone cannot see the magnitude. Check it on the samples themselves:")

good_fit = D < np.quantile(D, 0.05)  # the 5 % that fit the orientations best
print(f"    among the 5 % best-fitting samples ({good_fit.sum()}), the magnitude")
print(f"    spans {MG[good_fit].min()/1e6:.1f} .. {MG[good_fit].max()/1e6:.1f} MPa "
      f"— essentially the whole prior "
      f"({MG.min()/1e6:.1f} .. {MG.max()/1e6:.1f} MPa).")

adm = P <= 1e-12
print(f"\n  The Coulomb criterion IS scale-dependent (the lithostatic mode pins the")
print(f"  absolute stress level), so it bounds the magnitude from above:")
print(f"    frictionally admissible samples (PI = 0): {adm.sum()}/{N_SAMPLES}")
if adm.any():
    print(f"    their magnitude spans {MG[adm].min()/1e6:.1f} .. "
          f"{MG[adm].max()/1e6:.1f} MPa   (truth {MAG_TRUE/1e6:.0f} MPa)")
    both = good_fit & adm
    if both.any():
        print(f"    fitting the data AND admissible: {both.sum()} samples, "
              f"magnitude {MG[both].min()/1e6:.1f} .. {MG[both].max()/1e6:.1f} MPa")

# --- 6c. posteriors: what each cost function actually constrains --------------
keep_data = J_data <= np.quantile(J_data, 0.02)  # best 2 % by data misfit
keep_full = J_full <= np.quantile(J_full, 0.02)  # best 2 % by data + penalty

print(f"\n  Posterior (best 2 % = {keep_data.sum()} samples), mean +/- std:")
print(f"  {'':12s} {'data only':>22s} {'data + Coulomb':>22s} {'truth':>10s}")
for nm, col, scale, unit in (
    ("theta", TH, 1.0, "deg"),
    ("R", RR, 1.0, ""),
    ("magnitude", MG, 1e-6, "MPa"),
):
    a, b = col[keep_data] * scale, col[keep_full] * scale
    tru = {"theta": THETA_TRUE, "R": R_TRUE, "magnitude": MAG_TRUE * 1e-6}[nm]
    print(
        f"  {nm:12s} {a.mean():10.2f} +/- {a.std():6.2f} "
        f"{b.mean():13.2f} +/- {b.std():6.2f} {tru:10.2f} {unit}"
    )
print("\n  theta is resolved by the orientation data alone; R only loosely (the")
print("  misfit is a shallow function of R); the magnitude is resolved ONLY once")
print("  the Coulomb term is included.")

# --- 6d. diagnostics at the recovered solution -------------------------------
a_best = alpha_of(TH[best_full], RR[best_full], MG[best_full])
ev = np.array(cs.evaluateCoulomb(a_best).asArray()).reshape(-1, 4)
print(f"\n  Fault activity predicted at the recovered stress state "
      f"(PI = {cs.coulombPenalty(a_best):.4f}):")
off = 0
for name, s in zip(fault_names, faults):
    n = s.nbTriangles()
    sl = slice(off, off + n)
    act = ev[sl, 1] > 0
    line = f"    {name:10s} {100.0*act.mean():5.1f} % of elements sliding"
    if act.any():
        m, sd, _ = rake_stats(ev[sl, 2][act], ev[sl, 3][act])
        line += (f", slip axis {m:6.1f} +/- {sd:.0f} deg, "
                 f"max excess {ev[sl, 0].max()/1e6:.2f} MPa")
    else:
        line += "  (sub-critical: the whole fault is below its frictional strength)"
    print(line)
    off += n

# The same state evaluated at the reference magnitude of section 4, to show what
# an over-pressurised / more strongly loaded version of the same solution does.
a_hot = alpha_of(TH[best_full], RR[best_full], MAG0)
ev_hot = np.array(cs.evaluateCoulomb(a_hot).asArray()).reshape(-1, 4)
print(f"\n  Same orientation, magnitude raised to {MAG0/1e6:.0f} MPa: "
      f"{cs.nbSliding(a_hot)}/{n_elts} elements sliding, "
      f"PI = {cs.coulombPenalty(a_hot):.4f}")
print(f"    -> this is the upper bound the criterion places on the loading.")

rule()
print("Done.")
print(
    "\nWhat CoulombSuperposition produced here:\n"
    "  * an EXACT, ~5 orders of magnitude cheaper replacement for the BEM solve\n"
    "    inside the inversion loop (step 3);\n"
    "  * per-element Coulomb excess, sliding flag and rake for any candidate\n"
    "    stress state, i.e. a fault-activity prediction (steps 4 and 6d);\n"
    "  * a scalar, dimensionless admissibility penalty PI that maps the\n"
    "    mechanically viable region of parameter space (step 5);\n"
    "  * the extra constraint that resolves the stress MAGNITUDE, which\n"
    "    orientation data alone cannot determine (step 6b).\n"
    "\nRemember: PI is a fast screening filter evaluated on the *unrelaxed*\n"
    "tractions. Re-solve the preferred model with the Coulomb TIC (projected\n"
    "Gauss-Seidel) to obtain the physically consistent slip distribution.\n"
    "See arch3/docs/coulomb-superposition.md, section 7."
)
