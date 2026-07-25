#!/usr/bin/env python3
# ============================================================================
# Hector Mine (1999, Mw 7.1) slip inversion — python version of the C++ example
# main.cxx, using the rosetta-generated `arch` python binding.
#
# Replication of the historical Poly3Dinv run described in hector3.txt:
#
#   - model      : hector3flt.in (Poly3D deck, 3 segments, half-space,
#                  'ttb' elements: dip/strike free, opening locked)
#   - data       : dpqgrid.dat InSAR (x y z u), u scaled by 0.001,
#                  satellite line of sight (-0.38, 0.07, -0.92)
#   - smoothing  : 0.1
#   - constraints: slip pinned on the fault borders (except the free-surface
#                  trace and the segment intersections), negative dip-slip and
#                  negative strike-slip DICs
#   - output     : hector3flt_python_result.ts (Gocad TSurf, one object per
#                  segment, per-vertex properties BC1 BC2 BC3 like Poly3Dinv),
#                  compared against the Poly3Dinv reference
#                  hector3flt_result.ts when that file is present
#
# All the inputs are read from the directory of this script unless another one
# is given, and the binding is imported from ../../bindings/python-expanded.
#
# Usage: ./main.py [data_dir] [--smooth 0.1] [--max-iter 5000] [--eps 1e-10]
#
# ----------------------------------------------------------------------------
# API notes — how this differs from main.cxx
#
# The python binding exposes Model / Surface / SlipInsarData / SlipInversion
# but *not* Triangle, so everything main.cxx does element by element has to be
# expressed at the Surface level or through an inequality constraint:
#
#   1. Meshes are built with `model.addSurface(positions, indices)` (flat
#      lists), not `Surface(model, [Triangle*])`.
#   2. Boundary condition *types* are per surface: `surface.setBcType(axis,
#      type)` with axis in ("dip", "strike", "normal"). All the elements of the
#      deck share the same 'ttb' code, so this is enough here (the script
#      checks it and reports otherwise).
#   3. main.cxx pins the border elements by switching their bc type to
#      DISPLACEMENT (0 dof, so they leave the unknowns). Per-element bc types
#      are not reachable from python, so the borders are pinned with a
#      FunctionalDic that projects their slip back to zero at every iteration
#      of the projected Gauss-Seidel (SlipInversion calls applyDics() on each
#      element after its local solve). The pinned components stay at 0 exactly
#      as with a Dirichlet bc; the only difference is that they keep their
#      column in the least-squares system.
#   4. The same FunctionalDic carries the two "negative slip" constraints
#      (main.cxx uses a UserDic, which needs a Triangle).
#
# Conventions (this is where main.cxx gets it wrong, see below):
#   - arch's element-local axes are (normal, strike, dip) — Triangle.cxx builds
#     l2g_ = Matrix33(normal, strike, dip) — which is what the string API
#     ("normal" -> 0, "strike" -> 1, "dip" -> 2) uses.
#   - Poly3D orders them (dip, strike, normal) and its dip axis points the
#     other way (see Poly3DFilter::setInitBcValue, which negates it). So the
#     'ttb' code of the deck means dip free, strike free, *opening* locked, and
#     P3D's "negative for dip" is a *positive* arch dip component.
#     main.cxx maps the code straight onto the triangle axes (0 = dip), so it
#     locks the dip slip and lets the fault open instead — which is why
#     hector3flt_arch3_result.ts has a large BC1 (opening) and does not
#     look like the Poly3Dinv reference.
# ============================================================================

import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bindings", "python-expanded")
)

import arch  # noqa: E402


# Poly3D element-local components, in deck order, and how they map onto the
# arch element-local ones: (axis name, arch index, sign of the P3D component
# w.r.t. the arch component).
P3D_AXES = (("dip", 2, -1.0), ("strike", 1, 1.0), ("normal", 0, 1.0))

Z_TRACE = 0.5  # km: vertices on the free-surface trace have |z| ~ 1e-5
D_INTER = 0.1  # km: distance below which two segments are considered to touch


# ============================================================================
# 1. Poly3D deck
# ============================================================================
def parse_poly3d(filename):
    """Parse the Poly3D input file.

    Returns (constants, vertices, segments) where vertices is a list of
    (x, y, z) and each segment is a dict with its name, its triangles (global
    vertex ids), and the per-element bc codes/values.
    """
    constants = {}
    vertices = []
    vertex_id = {}
    segments = []

    with open(filename) as f:
        for line in f:
            toks = line.split()
            if not toks or toks[0][0] in "*#":
                continue

            key = toks[0]

            if len(toks) >= 3 and toks[1] == "=":
                constants[key] = toks[2]
            elif key == "v" and len(toks) >= 6:
                # v <name> <csys> <x> <y> <z>
                vertex_id[toks[1]] = len(vertices)
                vertices.append((float(toks[3]), float(toks[4]), float(toks[5])))
            elif key == "o" and len(toks) >= 2:
                # o "<name>" ...
                segments.append(
                    {"name": toks[1].replace('"', ""), "tris": [], "codes": [], "values": []}
                )
            elif key == "e" and len(toks) >= 10:
                # e 3 <csys> <bc code> <bc1> <bc2> <bc3> <v1> <v2> <v3>
                if toks[1] != "3":
                    raise ValueError("only 3-node elements are supported")
                if not segments:
                    raise ValueError("element before any object")
                seg = segments[-1]
                seg["tris"].append(tuple(vertex_id[n] for n in toks[7:10]))
                seg["codes"].append(toks[3])
                seg["values"].append([float(v) for v in toks[4:7]])

    return constants, vertices, segments


def build_surfaces(model, vertices, segments):
    """One arch Surface per Poly3D object, with the deck boundary conditions."""
    for seg in segments:
        # local (per surface) vertex numbering, in order of first appearance
        local = {}
        order = []
        indices = []
        for tri in seg["tris"]:
            for g in tri:
                if g not in local:
                    local[g] = len(order)
                    order.append(g)
                indices.append(local[g])

        positions = [c for g in order for c in vertices[g]]
        seg["local"] = local
        seg["order"] = order
        seg["surface"] = model.addSurface(positions, indices)

        # Boundary conditions. Only per-surface bc types are exposed, so the
        # whole object must share one code ('t' = traction/free = unknown
        # Burgers component, 'b' = prescribed Burgers component).
        codes = set(seg["codes"])
        if len(codes) != 1:
            raise ValueError(
                f"segment {seg['name']}: mixed element bc codes {sorted(codes)}; the python "
                "binding only exposes per-surface bc types"
            )
        code = codes.pop()

        for a, (axis, _, sign) in enumerate(P3D_AXES):
            seg["surface"].setBcType(axis, "free" if code[a] == "t" else "fixed")
            column = [sign * v[a] for v in seg["values"]]
            if any(v != 0.0 for v in column):
                seg["surface"].setBcValues(axis, column)

        print(
            f"segment {seg['name']}: {seg['surface'].nbTriangles()} elements, "
            f"{seg['surface'].nbVertices()} vertices, bc '{code}'"
        )


# ============================================================================
# 2. Constraints
# ============================================================================
def border_triangles(seg):
    """Indices of the elements of a segment having less than 3 edge neighbors."""
    count = {}
    for tri in seg["tris"]:
        for a in range(3):
            edge = (min(tri[a], tri[(a + 1) % 3]), max(tri[a], tri[(a + 1) % 3]))
            count[edge] = count.get(edge, 0) + 1

    borders = []
    for k, tri in enumerate(seg["tris"]):
        neighbors = 0
        for a in range(3):
            edge = (min(tri[a], tri[(a + 1) % 3]), max(tri[a], tri[(a + 1) % 3]))
            neighbors += count[edge] - 1
        if neighbors < 3:
            borders.append(k)
    return borders


def pinned_elements(segments, vertices):
    """Poly3Dinv "scale 0.0001 on border for slip": pin the slip on the border
    elements, except along the free-surface trace (the fault ruptured the
    surface) and near the segment intersections ("scale 1 on intersection").

    Returns, per segment, the set of pinned element indices.
    """
    pinned = []
    for s, seg in enumerate(segments):
        others = [v for o, other in enumerate(segments) if o != s for tri in other["tris"] for v in tri]
        others = set(others)

        keep = set()
        for k in border_triangles(seg):
            tri = seg["tris"][k]

            # free-surface trace: an edge (2 vertices) at z ~ 0
            if sum(1 for g in tri if abs(vertices[g][2]) < Z_TRACE) >= 2:
                continue

            # segment intersection: a vertex close to another segment
            touching = False
            for g in tri:
                x, y, z = vertices[g]
                for h in others:
                    ox, oy, oz = vertices[h]
                    if math.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2) < D_INTER:
                        touching = True
                        break
                if touching:
                    break
            if touching:
                continue

            keep.add(k)
        pinned.append(keep)

    print(f"pinned {sum(len(p) for p in pinned)} border elements")
    return pinned


def add_constraints(segments, vertices, pinned):
    """One FunctionalDic per segment: zero slip on the pinned border elements,
    negative dip slip and negative strike slip everywhere else.

    A FunctionalDic is called with the element center and its current slip, both
    in the arch element-local frame (normal, strike, dip), and returns the
    projected slip. Elements are identified by their center, quantized to 0.1 mm
    (the elements here are kilometers wide).
    """
    dics = []
    for seg, keep in zip(segments, pinned):
        centers = set()
        for k in keep:
            tri = seg["tris"][k]
            centers.add(
                tuple(
                    round(sum(vertices[g][c] for g in tri) / 3.0, 4) for c in range(3)
                )
            )

        # P3D asks for dip <= 0 and strike <= 0; the arch dip axis is the
        # opposite of the P3D one (P3D_AXES), hence the max() on the dip.
        def project(at, displ, centers=centers):
            if (round(at[0], 4), round(at[1], 4), round(at[2], 4)) in centers:
                return [0.0, 0.0, 0.0]
            return [displ[0], min(displ[1], 0.0), max(displ[2], 0.0)]

        dic = arch.FunctionalDic()
        dic.setFunction(project)
        dic.addTo(seg["surface"])
        dics.append(dic)  # the binding takes a raw pointer: keep it alive

    return dics


# ============================================================================
# 3. InSAR dataset
# ============================================================================
def read_insar(filename, scale):
    """dpqgrid.dat: 'x y z u' per line, u scaled by scale_u."""
    pos, u = [], []
    with open(filename) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 4:
                continue
            pos.extend(float(v) for v in toks[0:3])
            u.append(scale * float(toks[3]))
    return pos, u


# ============================================================================
# 5. Gocad TSurf output
# ============================================================================
def segment_slip(seg):
    """Per-vertex slip of a segment, in Poly3D component order (BC1 BC2 BC3),
    averaged from the adjacent elements (same layout as Poly3Dinv)."""
    displ = seg["surface"].displ(True, True)  # local=True, atTriangles=True
    slip = [[0.0, 0.0, 0.0] for _ in seg["order"]]
    count = [0] * len(seg["order"])

    for k, tri in enumerate(seg["tris"]):
        d = displ[3 * k : 3 * k + 3]
        for g in tri:
            l = seg["local"][g]
            for a, (_, index, sign) in enumerate(P3D_AXES):
                slip[l][a] += sign * d[index]
            count[l] += 1

    return [[c / n for c in s] for s, n in zip(slip, count)]


def write_result(filename, segments, vertices):
    with open(filename, "w") as o:
        for seg in segments:
            slip = segment_slip(seg)
            o.write("GOCAD TSurf 1.0\n")
            o.write("HEADER {\nname: %s\n}\n" % seg["name"])
            o.write("PROPERTIES BC1 BC2 BC3\n\n")
            for l, g in enumerate(seg["order"]):
                x, y, z = vertices[g]
                o.write(
                    "PVRTX %d %g %g %g %g %g %g\n" % (l, x, y, z, slip[l][0], slip[l][1], slip[l][2])
                )
            for tri in seg["tris"]:
                o.write("TRGL %d %d %d\n" % tuple(seg["local"][g] for g in tri))
            o.write("END\n")
    print(f"result written to {filename}")


def report_misfit(model, pos, u, los):
    """Line-of-sight rms misfit of the inverted slip against the InSAR data.

    Postprocess evaluates the displacement induced by the slip currently held by
    the elements, i.e. the solution, so no extra bookkeeping is needed.
    """
    displ = arch.Postprocess(model).displ(pos)
    data = residual = 0.0
    n = 0
    for k in range(len(u)):
        d = displ[3 * k : 3 * k + 3]
        if not all(math.isfinite(v) for v in d):
            continue  # a data point sitting on the fault trace: singular kernel
        predicted = sum(los[i] * d[i] for i in range(3))
        data += u[k] ** 2
        residual += (predicted - u[k]) ** 2
        n += 1

    data, residual = math.sqrt(data / n), math.sqrt(residual / n)
    print(f"\n--- InSAR misfit ({n} points) ---")
    print(f"  data rms     = {data:.4e}")
    print(f"  residual rms = {residual:.4e}  ({100 * (1 - residual / data):.1f}% reduction)")


def read_tsurf(filename):
    """{segment name: {(x,y,z): (BC1,BC2,BC3)}} of a Gocad TSurf result."""
    result, name = {}, None
    with open(filename) as f:
        for line in f:
            toks = line.split()
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
                result[name] = {}
            elif toks and toks[0] == "PVRTX" and len(toks) >= 8:
                key = tuple(round(float(v), 3) for v in toks[2:5])
                result[name][key] = tuple(float(v) for v in toks[5:8])
    return result


def compare(filename, segments, vertices):
    """Per-component RMS difference against the Poly3Dinv reference result."""
    reference = read_tsurf(filename)
    print(f"\n--- Comparison with the Poly3Dinv reference ({os.path.basename(filename)}) ---")
    print(f"  {'segment':>8} {'component':>10} {'this run':>12} {'reference':>12} {'rms diff':>12}")
    for seg in segments:
        if seg["name"] not in reference:
            print(f"  {seg['name']:>8}: not in the reference")
            continue
        ref = reference[seg["name"]]
        slip = segment_slip(seg)
        for a in range(3):
            n, sq, mine, theirs = 0, 0.0, 0.0, 0.0
            for l, g in enumerate(seg["order"]):
                key = tuple(round(c, 3) for c in vertices[g])
                if key not in ref:
                    continue
                n += 1
                sq += (slip[l][a] - ref[key][a]) ** 2
                mine = max(mine, abs(slip[l][a]))
                theirs = max(theirs, abs(ref[key][a]))
            if n == 0:
                print(f"  {seg['name']:>8} BC{a + 1:<9} no matching vertex")
                continue
            print(
                f"  {seg['name']:>8} {'BC%d' % (a + 1):>10} {mine:12.4e} {theirs:12.4e} "
                f"{math.sqrt(sq / n):12.4e}   (max |.| over {n} vertices)"
            )


# ============================================================================
# main
# ============================================================================
def main(argv):
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Hector Mine slip inversion")
    parser.add_argument("directory", nargs="?", default=here, help="where the deck and the data are")
    # Poly3Dinv's "smooth .1". Note that arch3 normalizes the Tikhonov weight
    # differently (lambda = eps^2 tr(GtWG)/tr(LtL)), so 0.1 here is not the 0.1
    # of the historical run: it gives a rougher solution, with a peak slip about
    # ten times the one of hector3flt_result.ts and, in practice, a larger InSAR
    # residual than e.g. --smooth 1 (54% vs 86% rms reduction).
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--eps", type=float, default=1e-10)
    args = parser.parse_args(argv[1:])
    directory = args.directory

    # 1. Model: 3 fault segments in an elastic half-space
    constants, vertices, segments = parse_poly3d(os.path.join(directory, "hector3flt.in"))

    model = arch.Model()
    model.setHalfSpace(constants.get("half_space", "yes") == "yes")
    poisson = float(constants.get("psn_ratio", 0.25))
    young = float(constants.get("youngs_mod", 1.0))
    # The displacement kernels of a dislocation only depend on the Poisson
    # ratio, so young/density have no influence on this inversion.
    model.setMaterial(poisson, young, float(constants.get("density", 0.0)))
    print(f"half space: {model.halfSpace()}, poisson: {poisson}, young: {young}")

    build_surfaces(model, vertices, segments)

    # 2. Constraints: slip pinned on the buried fault borders, negative dip
    #    slip and negative strike slip (DICs)
    dics = add_constraints(segments, vertices, pinned_elements(segments, vertices))

    # 3. InSAR dataset (scale_u 0.001, satellite -0.38 0.07 -0.92)
    los = (-0.38, 0.07, -0.92)
    pos, u = read_insar(os.path.join(directory, "dpqgrid.dat"), 0.001)
    print(f"{len(u)} InSAR points")
    insar = arch.SlipInsarData(pos, u, arch.Vector3(*los))

    # 4. Invert (smooth 0.1)
    print(f"\n--- Inverting (smooth {args.smooth}) ---")
    inversion = arch.SlipInversion(model)
    inversion.addData(insar, 1.0)
    inversion.setRegularization(args.smooth)
    inversion.setMaxIter(args.max_iter)
    inversion.setEps(args.eps)
    inversion.solve()

    # 5. Report, write the Gocad TSurf result and compare with Poly3Dinv
    print("\n--- Slip (Poly3D components) ---")
    for seg in segments:
        slip = segment_slip(seg)
        for a, (axis, _, _) in enumerate(P3D_AXES):
            values = [s[a] for s in slip]
            print(
                f"  {seg['name']:>5} {axis:>7}: min = {min(values):12.4e}, max = {max(values):12.4e}"
            )

    report_misfit(model, pos, u, los)
    write_result(os.path.join(directory, "hector3flt_python_result.ts"), segments, vertices)

    reference = os.path.join(directory, "hector3flt_result.ts")
    if os.path.exists(reference):
        compare(reference, segments, vertices)

    assert dics  # they must outlive solve()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
