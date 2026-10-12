#!/usr/bin/env python3
"""The five synthetic tests of the friction-inversion paper, recomputed with arch.

One non-planar, irregularly bounded fault (dip ~60 deg, corrugated, with a steeper
flexure).  A friction distribution (and, in test 5, a cohesion) is imposed with arch's
Coulomb (isotropic) or CoulombOrtho (friction and cohesion along the strike and dip
axes, rectangular law).  The
forward model gives the slip; the slip is smoothed by a Gaussian of radius BLUR to
mimic a regularised slip inversion; FrictionInversion is run with the true remote stress.

  1  heterogeneous isotropic friction (letters; bands)
  2  heterogeneous anisotropic friction (bands along dip, letters along strike)
  3  isotropic friction inverted as anisotropic
  4  anisotropic friction inverted as isotropic
  5  heterogeneous friction and cohesion (anisotropic), nodal least-squares fit

    python3 examples/fricinv/synthetic_tests.py [--blur 600] [--out DIR]
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import Delaunay
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

import common as K

arch = K.arch
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- geometry
L, W, H = 14000.0, 9000.0, 380.0           # along strike, down dip, mesh size (m)
STRIKE, DIP, TOP = 0.0, 60.0, -800.0
AMP, FLEX = 120.0, 400.0                   # corrugation and flexure amplitudes (m)
STRESS = dict(H=-85e6, h=-10e6, v=-130e6, theta=30.0)   # oblique: shear/normal >= 0.21 in both components everywhere


def boundary(u, v):
    """Inside test of the irregular outline, in (u, v) plane coordinates."""
    x, y = u / (0.5 * L), (v - 0.5 * W) / (0.5 * W)
    th = np.arctan2(y, x)
    r = 1.0 + 0.07 * np.sin(3 * th + 0.4) + 0.05 * np.cos(5 * th) - 0.06 * np.sin(2 * th)
    return x * x + y * y <= r * r * 0.96


def frame():
    s, d = np.radians(STRIKE), np.radians(DIP)
    us = np.array([np.sin(s), np.cos(s), 0.0])
    ud = np.array([np.cos(s) * np.cos(d), -np.sin(s) * np.cos(d), -np.sin(d)])
    return us, ud, np.cross(us, ud)


def surface_point(u, v):
    us, ud, n0 = frame()
    w = AMP * np.sin(2 * np.pi * u / 6000.0) * (0.6 + 0.4 * v / W) \
        + FLEX * np.tanh((u - 3500.0) / 900.0)                      # corrugation + flexure
    return np.array([0.0, 0.0, TOP]) + u[:, None] * us + v[:, None] * ud + w[:, None] * n0


def mesh(seed=3):
    rng = np.random.default_rng(seed)
    g = np.mgrid[-0.5 * L:0.5 * L:H, 0:W:H].reshape(2, -1).T
    g = g + rng.uniform(-0.25, 0.25, g.shape) * H
    g = g[boundary(g[:, 0], g[:, 1])]
    th = np.linspace(0, 2 * np.pi, int(2.4 * (L + W) / H), endpoint=False)          # outline points
    ring = []
    for t in th:                                           # bisect the radius of the outline
        lo, hi = 0.0, 1.3
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            u, v = mid * 0.5 * L * np.cos(t), 0.5 * W + mid * 0.5 * W * np.sin(t)
            lo, hi = (mid, hi) if boundary(np.array([u]), np.array([v]))[0] else (lo, mid)
        ring.append((lo * 0.5 * L * np.cos(t), 0.5 * W + lo * 0.5 * W * np.sin(t)))
    ring = np.array(ring)
    keep = np.min(np.linalg.norm(g[:, None] - ring[None], axis=2), axis=1) > 0.45 * H
    uv = np.vstack([g[keep], ring])
    tri = Delaunay(uv).simplices
    c = uv[tri].mean(axis=1)
    tri = tri[boundary(c[:, 0], c[:, 1])]
    P = surface_point(uv[:, 0], uv[:, 1])
    a, b, cc = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    n = np.cross(b - a, cc - a)
    flip = n @ frame()[2] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return uv, P, tri


# ----------------------------------------------------------------------------- patterns
def letters(uv_c, text="FA"):
    tp = TextPath((0, 0), text, size=1.0, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    ext = tp.get_extents()
    sx = 0.62 * L / ext.width
    x = (uv_c[:, 0] + 0.31 * L) / sx + ext.x0
    y = (W - uv_c[:, 1] - 0.5 * W) / sx + ext.y0 + 0.5 * ext.height
    return tp.contains_points(np.column_stack([x, y])).astype(float)


def bands(uv_c):
    u, v = uv_c[:, 0], uv_c[:, 1]
    return 0.5 * (1 + np.sin(2 * np.pi * (v + 900.0 * np.sin(2 * np.pi * u / 7000.0)) / 3000.0))


def smooth(C, val, r):
    if r <= 0:
        return val.copy()
    out = np.empty_like(val)
    for i, c in enumerate(C):
        w = np.exp(-np.sum((C - c) ** 2, axis=1) / (2 * r * r))
        out[i] = w @ val / w.sum()
    return out


# ----------------------------------------------------------------------------- forward / inversion
def build_model(P, tri, keep):
    m = arch.Model(); m.setHalfSpace(True); m.setMaterial(K.POISSON, K.YOUNG, K.DENSITY)
    s = m.addSurface(P.ravel().tolist(), tri.ravel().tolist())
    for nm in ("dip", "strike", "normal"):
        s.setBcType(nm, "free")
    return m, s


def forward(P, tri, mu_s, mu_d, c_s=None, c_d=None, solver="gmres"):
    """Coulomb friction: arch's Coulomb when the two axes share one friction (isotropic,
    circular law), CoulombOrtho otherwise (rectangular law, friction and cohesion given
    along the strike and dip axes)."""
    keep = []
    m, s = build_model(P, tri, keep)
    n = len(tri)
    cs = np.zeros(n) if c_s is None else np.asarray(c_s, float)
    cd = np.zeros(n) if c_d is None else np.asarray(c_d, float)
    if (mu_s is mu_d) and (c_d is None):
        tic = arch.Coulomb()
        tic.setFrictions(np.asarray(mu_s, float).tolist()); tic.setCohesions(cs.tolist())
    else:
        tic = arch.CoulombOrtho()                      # rectangular by default
        tic.setFrictionsStrike(np.asarray(mu_s, float).tolist()); tic.setFrictionsDip(np.asarray(mu_d, float).tolist())
        tic.setCohesionsStrike(cs.tolist()); tic.setCohesionsDip(cd.tolist())
    s.addTic(tic)
    m.addRemote(K.remote(keep, **STRESS)); keep.append(tic)
    if solver == "seidel":
        # The GMRES friction loop re-applies the TICs to its own constrained right-hand
        # side, so a cohesion is subtracted once per friction iteration (up to 9 c here).
        # Seidel recomputes the traction of each element at every sweep: cohesion once.
        sv = arch.SeidelSolver(m); sv.setEps(1e-13); sv.setMaxIter(200000); keep.append(sv)
        assert sv.run(False)
    else:
        assert K.solve(m, keep)
    return np.array(s.displ(True, True)).reshape(-1, 3)


def invert(P, tri, b, aniso):
    keep = []
    m, s = build_model(P, tri, keep)
    r = K.remote(keep, **STRESS); m.addRemote(r)
    inv = arch.FrictionInversion(m); ser = arch.Serie(3, b.ravel().tolist()); keep += [inv, ser]
    inv.setRemote(r); inv.setBurgers(ser); inv.useAnisotropy(aniso); inv.run()
    t0 = np.array(inv.tractions().asArray()).reshape(-1, 3)
    tr = np.array(inv.influences().asArray()).reshape(-1, 3)
    return np.array(inv.dipFriction().asArray()), np.array(inv.strikeFriction().asArray()), t0 + tr


def nodal_fit(tri, sig, tau, n_vertices, rings=1):
    """Coulomb line tau = mu sigma + c fitted by least squares over the elements around each
    vertex (exact normal equations: mu = cov/var); element value = mean of its vertices."""
    inc = [[] for _ in range(n_vertices)]
    for e, (a, b, c) in enumerate(tri):
        for v in (a, b, c):
            inc[v].append(e)
    nb = [set(x) for x in inc]
    for _ in range(rings - 1):                          # grow the stencil by element adjacency
        nb = [set().union(*(set(inc[v]) for e in s for v in tri[e])) for s in nb]
    mu_v = np.full(n_vertices, np.nan); c_v = np.full(n_vertices, np.nan); sd_v = np.zeros(n_vertices)
    for v, es in enumerate(nb):
        es = list(es)
        if len(es) < 3:
            continue
        x, y = sig[es], tau[es]
        var = x.var()
        sd_v[v] = np.sqrt(var)
        if var > 0:
            mu_v[v] = ((x * y).mean() - x.mean() * y.mean()) / var
            c_v[v] = y.mean() - mu_v[v] * x.mean()
    return np.nanmean(mu_v[tri], axis=1), np.nanmean(c_v[tri], axis=1), sd_v[tri].mean(axis=1)


def patch_fit(labels, sig, tau):
    """Coulomb line tau = mu sigma + c fitted by least squares over all the elements of each
    patch (labels: one integer per element).  Returns per-element mu, c and, per patch,
    the spread of sigma."""
    mu = np.full(len(sig), np.nan); c = np.full(len(sig), np.nan); spread = {}
    for lab in np.unique(labels):
        e = labels == lab
        x, y = sig[e], tau[e]
        var = x.var(); spread[int(lab)] = float(np.sqrt(var))
        m_ = ((x * y).mean() - x.mean() * y.mean()) / var
        mu[e] = m_; c[e] = y.mean() - m_ * x.mean()
    return mu, c, spread


# ----------------------------------------------------------------------------- main
def sweep(out):
    """Isotropic letters at two levels of criticality (friction as a fraction of the local slip
    tendency) and three slip smoothings: the error grows like (tau - mu sigma)/(mu sigma)."""
    uv, P, tri = mesh(); C = P[tri].mean(axis=1); uvc = uv[tri].mean(axis=1)
    t0 = invert(P, tri, np.zeros((len(tri), 3)), True)[2]
    rt = np.hypot(t0[:, 1], t0[:, 2]) / np.abs(t0[:, 0])
    FA = smooth(C, letters(uvc), 250.0)
    rows = []
    for lo, hi in ((0.30, 0.60), (0.60, 0.95)):
        mu = rt * (hi - (hi - lo) * FA)
        b = forward(P, tri, mu, mu)
        for blur in (0.0, 300.0, 600.0):
            d, _, _ = invert(P, tri, smooth(C, b, blur), False)
            rows.append(dict(fractions=[lo, hi], blur=blur, median_slip=float(np.median(np.linalg.norm(b[:, 1:], axis=1))),
                             rms=float(np.sqrt(np.mean((d - mu) ** 2))), corr=float(np.corrcoef(d, mu)[0, 1])))
            print(rows[-1])
    json.dump(rows, open(os.path.join(out, "synthetic_sweep.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur", type=float, default=300.0, help="Gaussian radius of the slip smoothing (m)")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--sweep", action="store_true", help="robustness to slip smoothing against criticality")
    a = ap.parse_args()
    if a.sweep:
        return sweep(a.out)
    uv, P, tri = mesh()
    C = P[tri].mean(axis=1); uvc = uv[tri].mean(axis=1)
    print(f"mesh: {len(P)} vertices, {len(tri)} triangles")
    # remote shear/normal ratios per element and component (slip tendency): friction is set as a
    # fraction of it, near-critical, where the inversion is robust to slip smoothing
    t0 = invert(P, tri, np.zeros((len(tri), 3)), True)[2]
    sn0 = np.abs(t0[:, 0]); rs, rd = np.abs(t0[:, 1]) / sn0, np.abs(t0[:, 2]) / sn0; rt = np.hypot(rs, rd)
    FA = smooth(C, letters(uvc), 250.0); BD = bands(uvc)
    hi, lo = 0.95, 0.60                                # fractions of the slip tendency
    frac = lambda pat: hi - (hi - lo) * pat            # low friction where the pattern is 1
    res = dict(uv=uv, P=P, tri=tri, uvc=uvc, rs=rs, rd=rd, FA=FA, BD=BD)
    stats = dict(blur=a.blur)

    def run(name, mu_s, mu_d, aniso, c_s=None, c_d=None, blur=a.blur):
        b = forward(P, tri, mu_s, mu_d, c_s, c_d)
        bs = smooth(C, b, blur)
        d, s, t = invert(P, tri, bs, aniso)
        res.update({f"{name}_mu_s": mu_s, f"{name}_mu_d": mu_d, f"{name}_rec_dip": d, f"{name}_rec_strike": s})
        stuck = np.mean((np.abs(b[:, 1]) < 1e-6) | (np.abs(b[:, 2]) < 1e-6))
        print(f"{name:10s} median |slip| {np.median(np.linalg.norm(b[:, 1:], axis=1)):.2f} m, stuck components {100 * stuck:.1f}%")
        return d, s, b

    r = lambda x, y: dict(rms=float(np.sqrt(np.mean((x - y) ** 2))), corr=float(np.corrcoef(x, y)[0, 1]))
    # 1: isotropic, circular law
    for name, pat in (("iso_FA", FA), ("iso_BD", BD)):
        mu = rt * frac(pat)
        d, _, _ = run(name, mu, mu, False)
        stats[name] = r(d, mu)
    # 2: anisotropic, rectangular law, bands along dip and letters along strike
    mus, mud = rs * frac(FA), rd * frac(BD)
    d, s, _ = run("aniso", mus, mud, True)
    stats["aniso"] = dict(dip=r(d, mud), strike=r(s, mus))
    # 3: isotropic friction read component by component: projections mu |cos|, mu |sin| of the rake
    mu = rt * frac(FA)
    d, s, b = run("isotrue", mu, mu, True)
    stats["isotrue"] = dict(hypot=r(np.hypot(d, s), mu), dip_vs_strike=r(d, s))
    # 4: anisotropic friction read as isotropic: hypot(mu_s, mu_d) where both components slide
    d, _, _ = run("anisotrue", mus, mud, False)
    stats["anisotrue"] = dict(hypot=r(d, np.hypot(mus, mud)), mean=r(d, 0.5 * (mus + mud)))
    # 5: friction and cohesion, exact slip, Seidel solver (cohesion applied once, see forward).
    # Piecewise-constant fields: friction 0.25 / 0.10 inside the letters FA, cohesion
    # 0 / 2 MPa inside the letter C, along strike; uniform 0.25 along dip.  Fitted (a) over
    # each patch of constant values, (b) over nodal stencils of 1-3 rings.
    inFA = smooth(C, letters(uvc), 250.0) > 0.5         # clean, filled masks: still piecewise constant
    inC = smooth(C, letters(uvc, "C"), 250.0) > 0.5
    mus5 = np.where(inFA, 0.10, 0.25); mud5 = np.full(len(C), 0.25)
    cs = np.where(inC, 2.0e6, 0.0); cd = np.zeros(len(C))
    b5 = forward(P, tri, mus5, mud5, cs, cd, solver="seidel")
    _, _, t5 = invert(P, tri, b5, True)
    sig = np.abs(t5[:, 0])
    labels = inFA.astype(int) + 2 * inC.astype(int)
    pm, pc, spread = patch_fit(labels, sig, np.abs(t5[:, 1]))
    res.update(coh_mu_s=mus5, coh_mu_d=mud5, coh_c_s=cs, coh_c_d=cd, coh_patch_mu_s=pm, coh_patch_c_s=pc,
               coh_labels=labels)
    stats["cohesion_patch"] = dict(mu_s_max_err=float(np.abs(pm - mus5).max()),
                                   c_s_max_err_MPa=float(np.abs(pc - cs).max() / 1e6),
                                   sigma_spread_MPa={k: v / 1e6 for k, v in spread.items()},
                                   stuck=float(np.mean((np.abs(b5[:, 1]) < 1e-9) | (np.abs(b5[:, 2]) < 1e-9))))
    for rings in (1, 2, 3):
        ms, c_s, sd = nodal_fit(tri, sig, np.abs(t5[:, 1]), len(P), rings)
        res.update({f"coh_r{rings}_mu_s": ms, f"coh_r{rings}_c_s": c_s, f"coh_r{rings}_sd": sd})
        stats[f"cohesion_ring{rings}"] = dict(mu_s=r(ms, mus5), c_s_MPa_rms=float(np.sqrt(np.mean((c_s - cs) ** 2)) / 1e6),
                                              sd_sigma_median_MPa=float(np.median(sd) / 1e6))
    np.savez(os.path.join(a.out, "synthetic_tests.npz"), **res)
    json.dump(dict(stress=STRESS, triangles=int(len(tri)), fractions=[lo, hi], stats=stats),
              open(os.path.join(a.out, "synthetic_tests.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
