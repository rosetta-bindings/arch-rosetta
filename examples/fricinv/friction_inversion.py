#!/usr/bin/env python3
"""Friction inversion on three interacting faults.

  1. Forward model with a heterogeneous Coulomb friction (0.10-0.35, every element
     sliding) under a strike-slip remote stress: slip b and surface GPS.
  2. FrictionInversion with b and the remote stress: the friction, element by element,
     isotropic and anisotropic (dip / strike components).
  3. Forward model again with the recovered friction: the GPS is reproduced.

The slip b stands for the output of a slip inversion; here it is exact, so the
recovered friction equals the imposed one to round-off.  See mech_vs_wb.py for what
happens when the stress itself has to be inverted.

    python3 examples/fricinv/friction_inversion.py      # -> friction_inversion.png
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import common as K

HERE = os.path.dirname(os.path.abspath(__file__))

C, N = K.geometry()
x, y, z = C.T
mu = 0.10 + 0.25 * 0.5 * (1 + np.sin(x / 2500.0 + y / 3000.0) * np.cos((z + 5000.0) / 2000.0))

keep = []
m, surfs, keep = K.model(frictions=mu, keep=keep)
assert K.solve(m, keep)
b = K.slip(surfs)
X = K.gps_grid()
u = K.gps(m, keep, X)
print(f"{len(mu)} triangles, mean |slip| {np.linalg.norm(b[:, 1:], axis=1).mean():.2f} m")

mu_iso, _ = K.friction_inversion(b, K.STRESS)
mu_dip, mu_strike = K.friction_inversion(b, K.STRESS, aniso=True)
print(f"isotropic:   max |error| {np.abs(mu_iso - mu).max():.2e}")
print(f"anisotropic: max |hypot(dip, strike) - mu| {np.abs(np.hypot(mu_dip, mu_strike) - mu).max():.2e}, "
      f"median dip / strike {np.median(mu_dip):.3f} / {np.median(mu_strike):.3f}")

keep2 = []
m2, s2, keep2 = K.model(frictions=mu_iso, keep=keep2)
assert K.solve(m2, keep2)
u2 = K.gps(m2, keep2, X)
print(f"GPS reproduced: rms {np.sqrt(np.mean((u2 - u) ** 2)):.2e} m of {np.sqrt(np.mean(u ** 2)):.2e} m")

# maps
geo = K.unfolded(); off = np.r_[0, np.cumsum([len(g[2]) for g in geo])]
rows = [(mu, "imposed"), (mu_iso, "recovered"), (mu_strike, "strike component"), (mu_dip, "dip component")]
fig, axs = plt.subplots(len(rows), 3, figsize=(8.0, 7.0), gridspec_kw=dict(hspace=0.35, wspace=0.1))
norm = mpl.colors.Normalize(0.0, 0.36)
for i, (v, lab) in enumerate(rows):
    for j, (s, d, I) in enumerate(geo):
        ax = axs[i, j]
        pc = ax.tripcolor(Triangulation(s, d, I), facecolors=v[off[j]:off[j + 1]], cmap="YlOrBr", norm=norm,
                          edgecolors="face", linewidth=0)
        ax.set_ylim(9.5, 0.5); ax.set_aspect("equal")
        if i == 0: ax.set_title(f"fault {j + 1}", fontsize=9)
        if j == 0: ax.set_ylabel(f"{lab}\ndepth (km)", fontsize=8)
        else: ax.set_yticklabels([])
        if i < len(rows) - 1: ax.set_xticklabels([])
        else: ax.set_xlabel("along strike (km)")
fig.colorbar(pc, ax=axs, shrink=0.5, label="friction")
out = os.path.join(HERE, "friction_inversion.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
