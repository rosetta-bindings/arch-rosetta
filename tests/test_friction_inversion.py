"""Tests of arch.FrictionInversion.

Truth: a forward model with Coulomb friction gives the slip b; FrictionInversion, given
b and the same remote stress, must return the imposed friction on every element that
slid (|t_s| = mu |t_n| there), a lower bound where an element stuck, and 0 where it opens.
"""
import numpy as np
import pytest

from conftest import (ONE_FAULT, THREE_FAULTS, STRESS, build, solve, slip, invert_friction, remote)

TOL = 1e-4


def forward(arch, faults, mu, stress=STRESS):
    keep = []
    m, surfs, C, keep = build(arch, faults, stress=stress, frictions=mu, keep=keep)
    assert solve(arch, m, keep)
    return slip(surfs), C, keep


def centres(faults):
    from conftest import rect_fault
    return np.concatenate([rect_fault(**f)[2] for f in faults])


def pattern(C, lo, hi):
    x, y, z = C.T
    u = 0.5 * (1 + np.sin(x / 2500.0 + y / 3000.0) * np.cos((z + 4000.0) / 2000.0))
    return lo + (hi - lo) * u


def test_uniform_friction_one_fault(arch):
    n = len(centres(ONE_FAULT))
    mu = np.full(n, 0.3)
    b, _, _ = forward(arch, ONE_FAULT, mu)
    assert np.all(np.linalg.norm(b[:, 1:], axis=1) > 0), "every element should slide"
    rec, _, _ = invert_friction(arch, ONE_FAULT, b)
    assert rec.shape == mu.shape
    assert np.abs(rec - mu).max() < TOL


def test_heterogeneous_friction_one_fault(arch):
    mu = pattern(centres(ONE_FAULT), 0.10, 0.35)
    b, _, _ = forward(arch, ONE_FAULT, mu)
    rec, _, _ = invert_friction(arch, ONE_FAULT, b)
    assert np.abs(rec - mu).max() < TOL


def test_several_surfaces(arch):
    """One result per triangle of every surface, and the slip of all surfaces enters."""
    mu = pattern(centres(THREE_FAULTS), 0.10, 0.35)
    b, _, _ = forward(arch, THREE_FAULTS, mu)
    rec, _, _ = invert_friction(arch, THREE_FAULTS, b)
    assert rec.size == mu.size
    assert np.abs(rec - mu).max() < TOL


def test_ignoring_the_other_surfaces_is_wrong(arch):
    """Inverting one fault alone with its own slip misses the traction induced by the
    others: the interaction is part of the measurement."""
    mu = pattern(centres(THREE_FAULTS), 0.10, 0.35)
    b, _, _ = forward(arch, THREE_FAULTS, mu)
    n0 = len(centres(ONE_FAULT))
    rec, _, _ = invert_friction(arch, ONE_FAULT, b[:n0])
    assert np.sqrt(np.mean((rec - mu[:n0]) ** 2)) > 10 * TOL


def test_isotropic_mode_strike_equals_dip(arch):
    mu = np.full(len(centres(ONE_FAULT)), 0.25)
    b, _, _ = forward(arch, ONE_FAULT, mu)
    dip, strike, _ = invert_friction(arch, ONE_FAULT, b, aniso=False)
    assert np.array_equal(dip, strike)


def test_anisotropic_components_of_an_isotropic_friction(arch):
    """For an isotropic friction the two components are the projections of one vector:
    hypot(mu_dip, mu_strike) = mu."""
    mu = pattern(centres(ONE_FAULT), 0.10, 0.35)
    b, _, _ = forward(arch, ONE_FAULT, mu)
    dip, strike, _ = invert_friction(arch, ONE_FAULT, b, aniso=True)
    assert np.abs(np.hypot(dip, strike) - mu).max() < TOL
    # strike-slip loading on a vertical fault: the strike component dominates
    assert np.median(strike) > np.median(dip)


def test_stuck_elements_give_a_lower_bound(arch):
    """Where the imposed friction exceeds what the loading can overcome the element sticks,
    and the inversion returns the friction it actually mobilised, below the imposed one."""
    C = centres(ONE_FAULT)
    mu = np.full(len(C), 0.2)
    high = np.abs(C[:, 1]) < 1500.0                   # a strong patch in the middle of the fault
    mu[high] = 5.0
    b, _, _ = forward(arch, ONE_FAULT, mu)
    rec, _, _ = invert_friction(arch, ONE_FAULT, b)
    assert np.all(rec[high] < mu[high])
    assert np.abs(rec[~high] - mu[~high]).max() < 1e-2


def test_opening_returns_zero(arch):
    """Under a tensile normal stress the element opens and no friction is defined."""
    tension = dict(H=10e6, h=20e6, v=5e6, theta=90.0)   # sigma_n > 0 on a N-S vertical fault
    n = len(centres(ONE_FAULT))
    rec, _, _ = invert_friction(arch, ONE_FAULT, np.zeros((n, 3)), stress=tension)
    assert np.all(rec == 0.0)


def test_wrong_stress_azimuth_degrades(arch):
    mu = pattern(centres(ONE_FAULT), 0.10, 0.35)
    b, _, _ = forward(arch, ONE_FAULT, mu)
    rec, _, _ = invert_friction(arch, ONE_FAULT, b, stress=dict(STRESS, theta=STRESS["theta"] + 5.0))
    assert np.sqrt(np.mean((rec - mu) ** 2)) > 0.01


def test_independent_traction_route(arch):
    """FrictionInversion against |t_s|/|t_n| computed from the stress on either side of each
    element, in a model whose displacements are imposed to b."""
    mu = pattern(centres(THREE_FAULTS), 0.10, 0.35)
    b, _, _ = forward(arch, THREE_FAULTS, mu)
    rec, _, _ = invert_friction(arch, THREE_FAULTS, b)
    from conftest import rect_fault, POISSON, YOUNG, DENSITY
    keep = []
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(POISSON, YOUNG, DENSITY)
    C, N, off = [], [], 0
    for f in THREE_FAULTS:
        pos, idx, c = rect_fault(**f)
        s = m.addSurface(pos, idx)
        for k, nm in enumerate(("normal", "strike", "dip")):
            s.setBcType(nm, "fixed"); s.setBcValues(nm, b[off:off + len(c), k].tolist())
        P = np.array(pos).reshape(-1, 3); I = np.array(idx).reshape(-1, 3)
        n = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
        N.append(n / np.linalg.norm(n, axis=1)[:, None]); C.append(c); off += len(c)
    m.addRemote(remote(arch, keep, **STRESS))
    assert solve(arch, m, keep)
    post = arch.Postprocess(m); post.setIncludeRemote(True); keep.append(post)
    C, N = np.concatenate(C), np.concatenate(N)
    mine = np.empty(len(C))
    for i, (c, n) in enumerate(zip(C, N)):
        t = 0.0
        for sgn in (1.0, -1.0):
            s = post.stressAt(*map(float, c + sgn * n))
            S = np.array([[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]])
            t = t + 0.5 * S @ n
        tn = t @ n
        mine[i] = np.linalg.norm(t - tn * n) / abs(tn)
    assert np.abs(rec - mine).max() < 1e-3


@pytest.mark.xfail(strict=True, reason="by design: Coulomb::apply returns early when the normal BC "
                                       "is a displacement ('fixed'), friction needs a traction normal BC")
def test_coulomb_acts_with_fixed_normal(arch):
    mu = np.full(len(centres(ONE_FAULT)), 0.3)
    keep = []
    m0, s0, _, keep = build(arch, ONE_FAULT, normal="fixed", keep=keep); solve(arch, m0, keep)
    m1, s1, _, keep = build(arch, ONE_FAULT, frictions=mu, normal="fixed", keep=keep); solve(arch, m1, keep)
    free = np.linalg.norm(slip(s0)[:, 1:], axis=1).mean()
    fric = np.linalg.norm(slip(s1)[:, 1:], axis=1).mean()
    assert fric < 0.9 * free
