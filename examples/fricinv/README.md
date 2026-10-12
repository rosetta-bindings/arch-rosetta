# Friction inversion

Examples of `arch.FrictionInversion` with the rosetta-generated binding
(`bindings/nanobind`, loaded by file path in `common.py`).

The friction inversion reads the friction from the traction that an observed slip
`b` leaves unbalanced under a remote stress, `t = sigma.n + T b`, as
`mu = |t_s| / |t_n|` on every element (`dipFriction()`; in anisotropic mode
`dipFriction()` = |t_dip / t_n| and `strikeFriction()` = |t_strike / t_n|).
It returns 0 where `t_n >= 0` (opening), and only a lower bound where an element
did not slide. The forward counterpart of the anisotropic reading is `CoulombOrtho`
(friction and cohesion along the strike and dip axes, `rectangular` law by default).

| script | what it does |
|---|---|
| `friction_inversion.py` | three interacting faults, heterogeneous Coulomb friction: forward model, friction inversion (isotropic and anisotropic), forward again with the recovered friction. Recovers the friction to round-off and writes `friction_inversion.png`. |
| `synthetic_tests.py`, `fig_synthetic.py` | the five synthetic tests of the friction-inversion paper on a non-planar fault: isotropic friction (`Coulomb`), anisotropic friction and cohesion along strike and dip (`CoulombOrtho`), slip smoothed to mimic a slip inversion, nodal friction-cohesion fit; `--sweep` for the robustness to slip smoothing against criticality. Writes `synthetic_tests.json/.npz` and `fig_synth_0..5.pdf`. |
| `mech_vs_wb.py` | which stress inversion can feed a friction inversion: the true stress, a frictionless mechanical inversion of the surface GPS (by superposition), and a Wallace-Bott inversion of the slip directions, each used whole (A), for its orientation only (B), or for its orientation and shape ratio (C). Writes `mech_vs_wb[_tag].json`. Companion of the paper *Which stress feeds a friction inversion?* |

```sh
python3 examples/fricinv/friction_inversion.py
python3 examples/fricinv/mech_vs_wb.py                         # mu 0.10-0.35
python3 examples/fricinv/mech_vs_wb.py --mu-lo 0.02 --mu-hi 0.12 --tag low
```

Tests: `python3 -m pytest tests -q` (`tests/test_friction_inversion.py`).

Notes
- Coulomb friction acts only on faults whose normal boundary condition is `free`;
  with `fixed` it is silently ignored (an `xfail` test records it).
- A stale `arch` installed in site-packages may shadow the fresh build; always load
  the binding by file path, as `common.py` and `tests/conftest.py` do.
