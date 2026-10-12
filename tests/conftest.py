"""Shared fixtures for the arch binding tests.

The binding is loaded by file path from ../bindings/nanobind (as in
examples/example_python.py): an editable install of `arch` puts a finder on
sys.meta_path that would otherwise pick up a stale site-packages build.

    python3 -m pytest tests -q
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pytest

_BINDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings", "nanobind")


def _load_arch():
    cand = sorted(glob.glob(os.path.join(_BINDING_DIR, "arch*.so")))
    if not cand:
        pytest.skip(f"no built `arch` extension in {_BINDING_DIR}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("arch", cand[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arch"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def arch():
    return _load_arch()


# ----------------------------------------------------------------------------- geometry
def rect_fault(center, strike, dip, length, top, bottom, h):
    """Rectangular fault meshed with triangles; strike in degrees from N, dip to the right.
    Returns (positions, indices) as flat lists, plus the (N,3) triangle centres."""
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
    return P.ravel().tolist(), idx, P[I].mean(axis=1)


# A small strike-slip set-up: faults at 20-40 deg to sigma_H, every element sliding for mu <= 0.35.
POISSON, YOUNG, DENSITY = 0.25, 30.0e9, 2700.0
STRESS = dict(H=-120e6, h=-50e6, v=-75e6, theta=20.0)          # compression negative, Pa
ONE_FAULT = [dict(center=(0.0, 0.0), strike=0.0, dip=90.0, length=8000.0, top=-1000.0, bottom=-7000.0, h=1000.0)]
THREE_FAULTS = ONE_FAULT + [
    dict(center=(5000.0, 2000.0), strike=40.0, dip=90.0, length=7000.0, top=-1000.0, bottom=-7000.0, h=1000.0),
    dict(center=(-5000.0, -3000.0), strike=150.0, dip=60.0, length=7000.0, top=-1000.0, bottom=-7000.0, h=1000.0)]


def remote(arch, keep, H, h, v, theta):
    r = arch.RemoteStress()
    r.setH(H); r.seth(h); r.setv(v); r.setTheta(theta)
    keep.append(r)
    return r


def build(arch, faults, stress=STRESS, frictions=None, normal="free", keep=None):
    """Model with free shear on every fault; frictions: per-triangle array (all faults
    concatenated) for a Coulomb TIC, or None.  Returns (model, surfaces, centres, keep)."""
    keep = [] if keep is None else keep
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(POISSON, YOUNG, DENSITY)
    surfs, C, off = [], [], 0
    for f in faults:
        pos, idx, c = rect_fault(**f)
        s = m.addSurface(pos, idx)
        s.setBcType("dip", "free"); s.setBcType("strike", "free"); s.setBcType("normal", normal)
        if normal == "fixed":
            s.setBcValue("normal", lambda x, y, z: 0.0)
        if frictions is not None:
            co = arch.Coulomb()
            co.setFrictions([float(v) for v in frictions[off:off + len(c)]]); co.setCohesion(0.0)
            s.addTic(co); keep.append(co)
        off += len(c)
        surfs.append(s); C.append(c)
    if stress is not None:
        m.addRemote(remote(arch, keep, **stress))
    return m, surfs, np.concatenate(C), keep


def solve(arch, m, keep):
    sv = arch.GmresSolver(m); sv.setCgnr(); sv.setEps(1e-10); sv.setMaxIter(5000)
    ok = sv.run(False); keep.append(sv)
    return ok


def slip(surfs):
    """Per-triangle displacement discontinuity in the local frame (normal, strike, dip)."""
    return np.concatenate([np.array(s.displ(True, True)).reshape(-1, 3) for s in surfs])


def invert_friction(arch, faults, b, stress=STRESS, aniso=False):
    """Run arch.FrictionInversion on a frictionless copy of the model with slip b."""
    keep = []
    m, _, _, keep = build(arch, faults, stress=None, keep=keep)
    r = remote(arch, keep, **stress); m.addRemote(r)
    inv = arch.FrictionInversion(m); keep.append(inv)
    ser = arch.Serie(3, b.ravel().tolist()); keep.append(ser)
    inv.setRemote(r); inv.setBurgers(ser); inv.useAnisotropy(aniso); inv.run()
    return np.array(inv.dipFriction().asArray()), np.array(inv.strikeFriction().asArray()), keep
