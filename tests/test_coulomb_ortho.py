"""Tests of arch.CoulombOrtho, friction and cohesion given along the dip and strike axes.

A planar fault dipping 60 deg under an oblique remote stress, so that both shear
components slide (shear/normal ratio above 0.4 on each axis).
"""
import numpy as np
import pytest

from conftest import rect_fault, remote, solve, slip, POISSON, YOUNG, DENSITY

DIPPING = [dict(center=(0.0, 0.0), strike=0.0, dip=60.0, length=8000.0, top=-1000.0, bottom=-7000.0, h=1000.0)]
OBLIQUE = dict(H=-85e6, h=-10e6, v=-130e6, theta=30.0)
TOL = 1e-4


def n_triangles():
    return len(rect_fault(**DIPPING[0])[2])


def forward(arch, tic_factory):
    """Free shear and normal (arch applies friction only with a traction normal BC)."""
    keep = []
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(POISSON, YOUNG, DENSITY)
    pos, idx, _ = rect_fault(**DIPPING[0])
    s = m.addSurface(pos, idx)
    for nm in ("dip", "strike", "normal"):
        s.setBcType(nm, "free")
    if tic_factory is not None:
        tic = tic_factory(); s.addTic(tic); keep.append(tic)
    m.addRemote(remote(arch, keep, **OBLIQUE))
    assert solve(arch, m, keep)
    return slip([s])


def invert(arch, b, aniso):
    keep = []
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(POISSON, YOUNG, DENSITY)
    pos, idx, _ = rect_fault(**DIPPING[0])
    s = m.addSurface(pos, idx)
    for nm in ("dip", "strike", "normal"):
        s.setBcType(nm, "free")
    r = remote(arch, keep, **OBLIQUE); m.addRemote(r)
    inv = arch.FrictionInversion(m); ser = arch.Serie(3, b.ravel().tolist()); keep += [inv, ser]
    inv.setRemote(r); inv.setBurgers(ser); inv.useAnisotropy(aniso); inv.run()
    return np.array(inv.dipFriction().asArray()), np.array(inv.strikeFriction().asArray())


def test_class_is_bound(arch):
    o = arch.CoulombOrtho()
    assert o.name() == "CoulombOrtho"


def test_elliptic_with_equal_axes_is_coulomb(arch):
    def coulomb():
        c = arch.Coulomb(); c.setFriction(0.3); c.setCohesion(0.0)
        return c

    def ortho():
        o = arch.CoulombOrtho(); o.setLaw("elliptic")
        o.setFrictionDip(0.3); o.setFrictionStrike(0.3)
        return o
    b0, b1 = forward(arch, coulomb), forward(arch, ortho)
    assert np.abs(b0 - b1).max() < 1e-8 * np.abs(b0).max()


def test_rectangular_recovered_by_friction_inversion(arch):
    """Different dip and strike frictions, per triangle: the anisotropic friction
    inversion returns both."""
    n = n_triangles()
    rng = np.random.default_rng(1)
    mu_d = rng.uniform(0.10, 0.30, n)
    mu_s = rng.uniform(0.05, 0.25, n)

    def ortho():
        o = arch.CoulombOrtho()                 # rectangular by default
        o.setFrictionsDip(mu_d.tolist()); o.setFrictionsStrike(mu_s.tolist())
        return o
    b = forward(arch, ortho)
    assert np.all(np.abs(b[:, 1]) > 0) and np.all(np.abs(b[:, 2]) > 0), "both components should slide"
    rec_d, rec_s = invert(arch, b, aniso=True)
    assert np.abs(rec_d - mu_d).max() < TOL
    assert np.abs(rec_s - mu_s).max() < TOL


def test_anisotropy_rotates_the_slip(arch):
    """A weak strike axis turns the slip towards strike compared with isotropic friction."""
    def ortho():
        o = arch.CoulombOrtho(); o.setFrictionDip(0.4); o.setFrictionStrike(0.05)
        return o

    def iso():
        c = arch.Coulomb(); c.setFriction(0.2)
        return c
    b_aniso, b_iso = forward(arch, ortho), forward(arch, iso)
    rake = lambda b: np.degrees(np.arctan2(np.abs(b[:, 2]), np.abs(b[:, 1])))
    assert np.median(rake(b_aniso)) < np.median(rake(b_iso)) - 5.0


def test_bad_law_raises(arch):
    o = arch.CoulombOrtho()
    with pytest.raises(Exception):
        o.setLaw("square")
