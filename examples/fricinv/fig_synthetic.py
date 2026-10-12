#!/usr/bin/env python3
"""3-D figures of the five synthetic tests (synthetic_tests.npz / .json).

    python3 examples/fricinv/fig_synthetic.py [--out DIR]     # fig_synth_1..5.pdf
"""
import argparse
import json
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator

import synthetic_tests as ST

HERE = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 8})
CMAP = "RdYlGn_r"                                   # green low, red high, as in the original figures


NLEV = 12                                           # number of iso-bands
_GEOM = {}


def nice_levels(norm):
    return mpl.ticker.MaxNLocator(nbins=NLEV, steps=[1, 2, 2.5, 5, 10]).tick_values(norm.vmin, norm.vmax)


def shifted(levels):
    """Levels nudged by a negligible amount, so that a field exactly equal to a round level
    (piecewise-constant data) does not produce contour lines from round-off noise."""
    return levels + 1e-4 * (levels[-1] - levels[0])


def set_geometry(uv, tri):
    """Refine the mesh once in the (u, v) plane of the fault; keep what the panels need."""
    T = Triangulation(uv[:, 0], uv[:, 1], tri)
    ref = UniformTriRefiner(T)
    Tr, idx = ref.refine_triangulation(return_tri_index=True, subdiv=3)
    Pr = ST.surface_point(Tr.x, Tr.y) / 1e3
    # outline: edges used by one triangle only
    cnt = {}
    for t in tri:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = tuple(sorted(e)); cnt[k] = cnt.get(k, 0) + 1
    edges = [k for k, c in cnt.items() if c == 1]
    Pv = ST.surface_point(uv[:, 0], uv[:, 1]) / 1e3
    _GEOM.update(T=T, ref=ref, Tr=Tr, Pr=Pr, tri=tri, nv=len(uv),
                 outline=[Pv[list(e)] for e in edges], lo=Pv.min(0), hi=Pv.max(0))


def to_vertices(val):
    """Element values -> vertex values (mean over the incident triangles)."""
    tri, nv = _GEOM["tri"], _GEOM["nv"]
    acc = np.zeros(nv); cnt = np.zeros(nv)
    for k in range(3):
        np.add.at(acc, tri[:, k], val); np.add.at(cnt, tri[:, k], 1.0)
    return acc / np.maximum(cnt, 1)


def panel(fig, spec, P, tri, val, norm, title, cmap=CMAP):
    """Filled iso-bands and iso-lines of an element field, drawn on the 3-D fault."""
    ax = fig.add_subplot(spec, projection="3d")
    vv = to_vertices(val)
    Tr, Pr = _GEOM["Tr"], _GEOM["Pr"]
    # linear interpolation: the default (cubic) overshoots and ripples on piecewise-constant data
    zr = _GEOM["ref"].refine_field(vv, triinterpolator=LinearTriInterpolator(_GEOM["T"], vv), subdiv=3)[1]
    levels = shifted(nice_levels(norm))
    bnorm = mpl.colors.BoundaryNorm(levels, plt.get_cmap(cmap).N, extend="both")
    fc = plt.get_cmap(cmap)(bnorm(zr[Tr.triangles].mean(axis=1)))
    ax.add_collection3d(Poly3DCollection(Pr[Tr.triangles], facecolors=fc, edgecolors="none"))
    # iso-lines, computed in (u, v) and mapped onto the surface
    tmp = plt.figure(); cs = tmp.add_subplot().tricontour(Tr, zr, levels=levels[1:-1]); plt.close(tmp)
    segs = []
    for lev in cs.allsegs:
        for seg in lev:
            if len(seg) > 1:
                segs.append(ST.surface_point(seg[:, 0], seg[:, 1]) / 1e3)
    ax.add_collection3d(Line3DCollection(segs, colors="#3A3A3A", linewidths=0.25), autolim=False)
    ax.add_collection3d(Line3DCollection(_GEOM["outline"], colors="#222222", linewidths=0.6), autolim=False)
    lo, hi = _GEOM["lo"], _GEOM["hi"]
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=30, azim=4)
    ax.set_title(title, fontsize=8, pad=-4)
    ax.set_ylabel("y (km)", labelpad=-8, fontsize=6.5); ax.set_zlabel("z (km)", labelpad=-9, fontsize=6.5)
    ax.set_xticks([])                                  # face-on: x adds nothing
    ax.tick_params(labelsize=5.5, pad=-3)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_alpha(0.0)
    return ax


def colorbar(fig, axes, norm, label, cmap=CMAP):
    levels = nice_levels(norm)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.BoundaryNorm(levels, plt.get_cmap(cmap).N, extend="both"), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.0, label=label, ticks=levels[::2])
    cb.ax.tick_params(labelsize=6.5)
    return cb


def grid(fig, n_rows, n_cols):
    return fig.add_gridspec(n_rows, n_cols, wspace=0.0, hspace=0.05)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=HERE); a = ap.parse_args()
    d = np.load(os.path.join(HERE, "synthetic_tests.npz"))
    st = json.load(open(os.path.join(HERE, "synthetic_tests.json")))["stats"]
    P, tri = d["P"], d["tri"]
    set_geometry(d["uv"], tri)

    def save(fig, k):
        fig.savefig(os.path.join(a.out, f"fig_synth_{k}.pdf"), bbox_inches="tight")
        plt.close(fig)

    def mnorm(*arrs):
        x = np.concatenate(arrs)
        return mpl.colors.Normalize(np.percentile(x, 1), np.percentile(x, 99))

    # 0 -- the surface, shaded by a low-angle light to express the relief
    T, ref = _GEOM["T"], _GEOM["ref"]
    Tr = ref.refine_triangulation(subdiv=2)
    Pr = ST.surface_point(Tr.x, Tr.y) / 1e3
    F = Pr[Tr.triangles]
    nrm = np.cross(F[:, 1] - F[:, 0], F[:, 2] - F[:, 0]); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nrm *= np.sign(nrm @ np.array([0.86, 0.06, 0.50]))[:, None]                    # towards the viewer side
    light = np.array([0.3, -0.85, 0.42]); light /= np.linalg.norm(light)
    dot = nrm @ light
    lo_, hi_ = np.percentile(dot, [2, 98])                                         # hillshade with contrast stretch
    shade = 0.30 + 0.70 * np.clip((dot - lo_) / (hi_ - lo_), 0, 1)
    base = np.array(mpl.colors.to_rgb("#B9C3D0"))
    fig = plt.figure(figsize=(5.2, 3.6))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(F, facecolors=np.clip(base[None] * shade[:, None], 0, 1), edgecolors="none"))
    ax.add_collection3d(Line3DCollection(_GEOM["outline"], colors="#222222", linewidths=0.7), autolim=False)
    lo, hi = _GEOM["lo"], _GEOM["hi"]
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(hi - lo)
    ax.view_init(elev=24, azim=-28)
    ax.set_xticks(np.round(np.linspace(lo[0], hi[0], 3), 1))
    ax.set_xlabel("x (km)", fontsize=7, labelpad=-2); ax.set_ylabel("y (km)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (km)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=-1)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_alpha(0.0)
    save(fig, 0)

    # 1 -- isotropic
    fig = plt.figure(figsize=(7.4, 4.6)); g = grid(fig, 2, 2); axs = []
    for i, (k, lab) in enumerate((("iso_FA", "letters"), ("iso_BD", "bands"))):
        n = mnorm(d[f"{k}_mu_s"], d[f"{k}_rec_dip"])
        axs += [panel(fig, g[i, 0], P, tri, d[f"{k}_mu_s"], n, f"imposed ({lab})"),
                panel(fig, g[i, 1], P, tri, d[f"{k}_rec_dip"], n,
                      f"recovered: rms {st[k]['rms']:.2f}, corr {st[k]['corr']:.2f}")]
        colorbar(fig, axs[-2:], n, r"friction $\mu$")
    save(fig, 1)

    # 2 -- anisotropic
    fig = plt.figure(figsize=(7.4, 4.6)); g = grid(fig, 2, 2)
    for i, (comp, key, lab) in enumerate((("d", "dip", r"$\mu_d$ (along dip)"), ("s", "strike", r"$\mu_s$ (along strike)"))):
        imp, rec = d[f"aniso_mu_{comp}"], d[f"aniso_rec_{key}"]
        n = mnorm(imp, rec); s = st["aniso"][key]
        ax = [panel(fig, g[i, 0], P, tri, imp, n, f"imposed {lab}"),
              panel(fig, g[i, 1], P, tri, rec, n, f"recovered: rms {s['rms']:.2f}, corr {s['corr']:.2f}")]
        colorbar(fig, ax, n, lab)
    save(fig, 2)

    # 3 -- isotropic friction read component by component
    imp = d["isotrue_mu_s"]; dd, ss = d["isotrue_rec_dip"], d["isotrue_rec_strike"]
    n = mnorm(imp, np.hypot(dd, ss), dd, ss)
    fig = plt.figure(figsize=(7.4, 4.6)); g = grid(fig, 2, 2)
    ax = [panel(fig, g[0, 0], P, tri, imp, n, r"imposed isotropic $\mu$"),
          panel(fig, g[0, 1], P, tri, np.hypot(dd, ss), n,
                rf"$\sqrt{{\mu_d^2+\mu_s^2}}$: corr {st['isotrue']['hypot']['corr']:.2f}"),
          panel(fig, g[1, 0], P, tri, dd, n, r"recovered $\mu_d = \mu\,|\cos\beta|$"),
          panel(fig, g[1, 1], P, tri, ss, n, r"recovered $\mu_s = \mu\,|\sin\beta|$")]
    colorbar(fig, ax, n, r"friction")
    save(fig, 3)

    # 4 -- anisotropic friction read as isotropic
    ms, md, rec = d["anisotrue_mu_s"], d["anisotrue_mu_d"], d["anisotrue_rec_dip"]
    n = mnorm(ms, md, np.hypot(ms, md), rec)
    fig = plt.figure(figsize=(7.4, 4.6)); g = grid(fig, 2, 2)
    s4 = st["anisotrue"]
    ax = [panel(fig, g[0, 0], P, tri, md, n, r"imposed $\mu_d$"),
          panel(fig, g[0, 1], P, tri, ms, n, r"imposed $\mu_s$"),
          panel(fig, g[1, 0], P, tri, np.hypot(ms, md), n, r"$\sqrt{\mu_s^2+\mu_d^2}$ of the imposed"),
          panel(fig, g[1, 1], P, tri, rec, n, f"recovered isotropic: corr {s4['hypot']['corr']:.2f}")]
    colorbar(fig, ax, n, r"friction")
    save(fig, 4)

    # 5 -- friction and cohesion: patch fit, against the nodal stencil
    fig = plt.figure(figsize=(7.4, 6.6)); g = grid(fig, 3, 2)
    sp = st["cohesion_patch"]; s1 = st["cohesion_ring1"]
    n = mpl.colors.Normalize(0.05, 0.30); nc = mpl.colors.Normalize(0.0, 2.5)
    ax1 = [panel(fig, g[0, 0], P, tri, d["coh_mu_s"], n, r"imposed $\mu_s$"),
           panel(fig, g[0, 1], P, tri, d["coh_patch_mu_s"], n,
                 rf"$\mu_s$ fitted per patch: max error {sp['mu_s_max_err']:.0e}")]
    colorbar(fig, ax1, n, r"friction $\mu_s$")
    ax2 = [panel(fig, g[1, 0], P, tri, d["coh_c_s"] / 1e6, nc, r"imposed cohesion $S_s$"),
           panel(fig, g[1, 1], P, tri, d["coh_patch_c_s"] / 1e6, nc,
                 rf"$S_s$ fitted per patch: max error {sp['c_s_max_err_MPa']:.0e} MPa")]
    colorbar(fig, ax2, nc, "cohesion (MPa)")
    ax3 = [panel(fig, g[2, 0], P, tri, np.clip(d["coh_r1_mu_s"], 0.05, 0.30), n,
                 rf"$\mu_s$ fitted per node (1 ring): corr {s1['mu_s']['corr']:.2f}")]
    sd = d["coh_r1_sd"] / 1e6
    ns = mpl.colors.Normalize(0, np.percentile(sd, 98))
    ax4 = [panel(fig, g[2, 1], P, tri, sd, ns, r"std$(\sigma_n)$ over the 1-ring stencil", cmap="viridis")]
    colorbar(fig, ax4, ns, r"std$(\sigma_n)$ (MPa)", cmap="viridis")
    save(fig, 5)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
