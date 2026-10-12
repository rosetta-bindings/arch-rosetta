#!/usr/bin/env python3
"""Which stress inversion can feed a friction inversion: mechanical or Wallace-Bott?

Truth: three interacting faults, heterogeneous friction, strike-slip remote stress
(common.py).  The frictional forward model gives the slip b (standing for a perfect slip
inversion) and noise-free surface GPS u.  Three stresses are fed to FrictionInversion:

  true   the true remote stress (control);
  mech   frictionless mechanical inversion of u: u is linear in (sxx, syy, sxy, szz)
         of an Andersonian tensor, four unit solutions and least squares;
  wb     Wallace-Bott: grid search over Andersonian reduced tensors minimising the
         mean angle between resolved shear and the slip direction of b; scale and
         isotropic part then fitted to u.

and each inverted stress is used (A) whole, (B) orientation only with the true
magnitudes, (C) orientation and shape ratio with sigma_v, sigma_h true.

    python3 examples/fricinv/mech_vs_wb.py [--mu-lo 0.10 --mu-hi 0.35]
"""
import argparse
import json
import os

import numpy as np

import common as K

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = [dict(H=1.0, h=0.0, v=0.0, theta=90.0),      # sxx
         dict(H=1.0, h=0.0, v=0.0, theta=0.0),       # syy
         dict(H=1.0, h=-1.0, v=0.0, theta=45.0),     # sxy
         dict(H=0.0, h=0.0, v=1.0, theta=0.0)]       # szz


def comps(sig):
    return np.array([sig[0, 0], sig[1, 1], sig[0, 1], sig[2, 2]])


def from_comps(c):
    return np.array([[c[0], c[2], 0.0], [c[2], c[1], 0.0], [0.0, 0.0, c[3]]])


def anderson(sig):
    r = K.to_remote(sig)
    vals = sorted([r["H"], r["h"], r["v"]])
    return dict(theta=r["theta"], R=(vals[1] - vals[2]) / (vals[0] - vals[2]), s1=vals[0], s2=vals[1], s3=vals[2])


def local_frame(N):
    st = np.cross(np.array([0, 0, 1.0]), N)
    st /= np.linalg.norm(st, axis=1)[:, None]
    return st, np.cross(N, st)


def wallace_bott(dirs, N):
    ev = np.array([0, 0, 1.0]); best = (np.inf, None)
    for regime in ("normal", "strike-slip", "thrust"):
        for thd in np.arange(0.0, 180.0, 1.0):
            th = np.radians(thd)
            eH = np.array([np.sin(th), np.cos(th), 0.0]); eh = np.array([np.cos(th), -np.sin(th), 0.0])
            e1, e2 = {"normal": (ev, eH), "strike-slip": (eH, ev), "thrust": (eH, eh)}[regime]
            A1, A2 = N @ e1, N @ e2
            for R in np.linspace(0.0, 1.0, 51):
                t = -(A1[:, None] * e1 + R * A2[:, None] * e2)
                tn = np.einsum("ij,ij->i", t, N); ts = t - tn[:, None] * N
                cs = np.einsum("ij,ij->i", ts, dirs) / np.maximum(np.linalg.norm(ts, axis=1), 1e-30)
                f = np.degrees(np.arccos(np.clip(cs, -1, 1))).mean()
                if f < best[0]:
                    best = (f, (regime, thd, R))
    regime, thd, R = best[1]; th = np.radians(thd)
    eH = np.array([np.sin(th), np.cos(th), 0.0]); eh = np.array([np.cos(th), -np.sin(th), 0.0])
    e1, e2 = {"normal": (ev, eH), "strike-slip": (eH, ev), "thrust": (eH, eh)}[regime]
    return -(np.outer(e1, e1) + R * np.outer(e2, e2)), dict(regime=regime, theta=thd, R=R, misfit=best[0])


def stats(rec, mu):
    e = rec - mu
    c = np.corrcoef(rec, mu)[0, 1] if mu.std() > 0 else float("nan")
    return dict(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)), corr=float(c),
                opening=float(np.mean(rec == 0.0)))      # FrictionInversion returns 0 where t_n >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-lo", type=float, default=0.10); ap.add_argument("--mu-hi", type=float, default=0.35)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    C, N = K.geometry(); x, y, z = C.T
    mu = a.mu_lo + (a.mu_hi - a.mu_lo) * 0.5 * (1 + np.sin(x / 2500.0 + y / 3000.0) * np.cos((z + 5000.0) / 2000.0))
    X = K.gps_grid()
    keep = []
    m, surfs, keep = K.model(frictions=mu, keep=keep); assert K.solve(m, keep)
    b = K.slip(surfs); u = K.gps(m, keep, X)

    G = []
    for par in UNITS:
        mk, _, keep = K.model(stress=par, keep=keep); K.solve(mk, keep); G.append(K.gps(mk, keep, X))
    G = np.array(G).T
    c, *_ = np.linalg.lstsq(G, u, rcond=None)
    sig_mech = from_comps(c)

    st, dp = local_frame(N)
    dirs = b[:, 1:2] * st + b[:, 2:3] * dp; dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    sig_red, wb = wallace_bott(dirs, N)
    Gs = np.column_stack([G @ comps(sig_red), G @ comps(np.eye(3))])
    (alpha, p), *_ = np.linalg.lstsq(Gs, u, rcond=None)
    sig_wb = alpha * sig_red + p * np.eye(3)

    true = K.STRESS
    out = dict(mu=[a.mu_lo, a.mu_hi], true=anderson(K.tensor(**true)), wb_search=wb)
    out["true_check"] = stats(K.friction_inversion(b, true)[0], mu)
    for name, sig in (("mech", sig_mech), ("wb", sig_wb)):
        an = anderson(sig)
        R = max(an["R"], 1e-3)
        cases = dict(A=K.to_remote(sig), B=dict(true, theta=an["theta"]),
                     C=dict(true, theta=an["theta"], H=true["h"] + (true["v"] - true["h"]) / R))
        out[name] = dict(stress=an, **{k: dict(remote=v, **stats(K.friction_inversion(b, v)[0], mu)) for k, v in cases.items()})
    json.dump(out, open(os.path.join(HERE, f"mech_vs_wb{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1)

    t = out["true"]
    print(f"true    azimuth N{t['theta']:05.1f}  R {t['R']:.2f}  s1 {t['s1'] / 1e6:7.1f} MPa   friction check rms {out['true_check']['rms']:.1e}")
    for name in ("mech", "wb"):
        s = out[name]["stress"]
        print(f"{name:7s} azimuth N{s['theta']:05.1f}  R {s['R']:.2f}  s1 {s['s1'] / 1e6:7.1f} MPa   "
              + "   ".join(f"{k}: rms {out[name][k]['rms']:.3f} (open {100 * out[name][k]['opening']:.0f}%)" for k in "ABC"))
    print(f"Wallace-Bott misfit {wb['misfit']:.2f} deg ({wb['regime']}, N{wb['theta']:05.1f}, R {wb['R']:.2f})")


if __name__ == "__main__":
    main()
