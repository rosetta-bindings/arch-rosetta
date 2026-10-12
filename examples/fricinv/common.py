"""Shared helpers for the friction-inversion examples: load the freshly built `arch`
binding by file path, build fault meshes and models, solve, read slip and run
FrictionInversion.  Compression negative, SI units."""
import glob
import importlib.util
import os
import sys

import numpy as np

_BINDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bindings", "nanobind")


def load_arch():
    cand = sorted(glob.glob(os.path.join(_BINDING_DIR, "arch*.so")))
    if not cand:
        sys.exit(f"No built `arch` extension found in {_BINDING_DIR} -- build it first.")
    spec = importlib.util.spec_from_file_location("arch", cand[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arch"] = mod
    spec.loader.exec_module(mod)
    return mod


arch = load_arch()

POISSON, YOUNG, DENSITY = 0.25, 30.0e9, 2700.0
STRESS = dict(H=-120e6, h=-50e6, v=-75e6, theta=20.0)      # effective, strike-slip, R = 0.36
FAULTS = [dict(center=(0.0, 0.0), strike=0.0, dip=90.0, length=12000.0),
          dict(center=(5000.0, 2000.0), strike=40.0, dip=90.0, length=10000.0),
          dict(center=(-5000.0, -3000.0), strike=150.0, dip=60.0, length=10000.0)]
TOP, BOTTOM, H = -1000.0, -9000.0, 500.0


def rect_fault(center, strike, dip, length, top=TOP, bottom=BOTTOM, h=H):
    """Rectangular fault meshed with triangles (strike from N, dip to the right).
    Returns flat positions, flat indices, triangle centres (N,3) and unit normals (N,3)."""
    s, d = np.radians(strike), np.radians(dip)
    us = np.array([np.sin(s), np.cos(s), 0.0])
    ud = np.array([np.cos(s) * np.cos(d), -np.sin(s) * np.cos(d), -np.sin(d)])
    width = (top - bottom) / np.sin(d)
    n1, n2 = max(2, int(round(length / h))), max(2, int(round(width / h)))
    p0 = np.array([center[0], center[1], top]) - 0.5 * length * us
    P = np.array([p0 + length * i / n1 * us + width * j / n2 * ud for j in range(n2 + 1) for i in range(n1 + 1)])
    idx = []
    for j in range(n2):
        for i in range(n1):
            v0 = j * (n1 + 1) + i
            v1, v2 = v0 + 1, v0 + n1 + 1
            v3 = v2 + 1
            idx += [v0, v1, v2, v1, v3, v2]
    I = np.array(idx).reshape(-1, 3)
    n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    return P.ravel().tolist(), idx, P[I].mean(axis=1), n / np.linalg.norm(n, axis=1)[:, None]


def geometry(faults=FAULTS):
    parts = [rect_fault(**f) for f in faults]
    return np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts])


def remote(keep, H, h, v, theta):
    r = arch.RemoteStress(); r.setH(H); r.seth(h); r.setv(v); r.setTheta(theta)
    keep.append(r)
    return r


def model(faults=FAULTS, stress=STRESS, frictions=None, keep=None):
    """Free shear and free normal on every fault (arch applies Coulomb only with a free
    normal); a Coulomb TIC with per-triangle frictions if given."""
    keep = [] if keep is None else keep
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(POISSON, YOUNG, DENSITY)
    surfs, off = [], 0
    for f in faults:
        pos, idx, c, _ = rect_fault(**f)
        s = m.addSurface(pos, idx)
        for nm in ("dip", "strike", "normal"):
            s.setBcType(nm, "free")
        if frictions is not None:
            co = arch.Coulomb(); co.setFrictions([float(v) for v in frictions[off:off + len(c)]]); co.setCohesion(0.0)
            s.addTic(co); keep.append(co)
        off += len(c)
        surfs.append(s)
    if stress is not None:
        m.addRemote(remote(keep, **stress))
    return m, surfs, keep


def solve(m, keep):
    sv = arch.GmresSolver(m); sv.setCgnr(); sv.setEps(1e-10); sv.setMaxIter(5000)
    ok = sv.run(False); keep.append(sv)
    return ok


def slip(surfs):
    """Per-triangle displacement discontinuity, local frame (normal, strike, dip)."""
    return np.concatenate([np.array(s.displ(True, True)).reshape(-1, 3) for s in surfs])


def gps(m, keep, X):
    post = arch.Postprocess(m); post.setIncludeRemote(False); keep.append(post)
    return np.array([post.displAt(*map(float, x)) for x in X]).ravel()


def gps_grid(n=25, L=15000.0):
    g = np.linspace(-L, L, n)
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def friction_inversion(b, stress, faults=FAULTS, aniso=False):
    """arch.FrictionInversion with slip b (local, per triangle) and a remote stress.
    Returns (mu_dip, mu_strike); in isotropic mode both hold the friction."""
    keep = []
    m, _, keep = model(faults, stress=None, keep=keep)
    r = remote(keep, **stress); m.addRemote(r)
    inv = arch.FrictionInversion(m); keep.append(inv)
    ser = arch.Serie(3, np.asarray(b).ravel().tolist()); keep.append(ser)
    inv.setRemote(r); inv.setBurgers(ser); inv.useAnisotropy(aniso); inv.run()
    return np.array(inv.dipFriction().asArray()), np.array(inv.strikeFriction().asArray())


def tensor(H, h, v, theta):
    th = np.radians(theta)
    eH = np.array([np.sin(th), np.cos(th), 0.0]); eh = np.array([np.cos(th), -np.sin(th), 0.0]); ev = np.array([0, 0, 1.0])
    return H * np.outer(eH, eH) + h * np.outer(eh, eh) + v * np.outer(ev, ev)


def to_remote(sig):
    """(H, h, v, theta) of a tensor with a vertical principal axis."""
    w, V = np.linalg.eigh(sig[:2, :2]); iH = int(np.argmin(w))
    return dict(H=float(w[iH]), h=float(w[1 - iH]), v=float(sig[2, 2]),
                theta=float(np.degrees(np.arctan2(V[0, iH], V[1, iH])) % 180.0))


def unfolded(faults=FAULTS):
    """Along-strike and depth coordinates (km) and triangles of each fault, for maps."""
    out = []
    for f in faults:
        pos, idx, _, _ = rect_fault(**f)
        P = np.array(pos).reshape(-1, 3)
        st = np.radians(f["strike"]); us = np.array([np.sin(st), np.cos(st), 0.0])
        out.append(((P - np.array([*f["center"], 0.0])) @ us / 1e3, -P[:, 2] / 1e3, np.array(idx).reshape(-1, 3)))
    return out
