#!/usr/bin/env python3
import glob
import importlib.util
import os
import sys

# Load the freshly built binding regardless of CWD.
#
# NOTE: a plain `import arch` is NOT enough: an editable (scikit-build) install
# of `arch` puts a finder on `sys.meta_path`, which is consulted BEFORE
# sys.path — so `import arch` would silently pick up the stale site-packages
# extension instead. Loading the .so by file path bypasses every finder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BINDING_DIR = os.path.join(_HERE, "..", "..", "bindings", "nanobind")
_candidates = sorted(glob.glob(os.path.join(_BINDING_DIR, "arch*.so")))
if _candidates:
    _spec = importlib.util.spec_from_file_location("arch", _candidates[0])
    arch = importlib.util.module_from_spec(_spec)
    sys.modules["arch"] = arch
    _spec.loader.exec_module(arch)
else:
    import arch  # fall back on the installed wheel


# ============================================================================
# 0. Gocad TSurf (.ts) reader
# ============================================================================
def load_gocad_ts(filename):
    """Read a Gocad TSurf ASCII file.

    Returns a list of parts, one per TFACE, each a dict:
        {"name": str, "positions": [x0,y0,z0, ...], "indices": [i0,i1,i2, ...]}

    `positions`/`indices` are the flat lists `Model.addSurface()` expects, with
    the triangle indices renumbered 0..n-1 *within each part* (Gocad vertex ids
    are file-global and may start at 0 or 1).

    Supported records: GOCAD TSurf, HEADER { name: ... }, TFACE, VRTX/PVRTX,
    ATOM/PATOM (vertex aliasing) and TRGL. Everything else (PROPERTIES, borders,
    BSTONE, ...) is ignored.
    """
    default_name = os.path.splitext(os.path.basename(filename))[0]
    # vertex id -> (x, y, z). Ids are shared by the TFACEs of one TSurf object
    # but restart from scratch at the next `GOCAD` line, so this is per object.
    coords = {}
    parts = []        # one entry per TFACE
    part = None
    obj_name = default_name  # HEADER of the TSurf currently being read
    in_header = False

    def new_part():
        # `order` keeps the vertices in declaration order so the part survives a
        # write/read round-trip unpermuted
        parts.append({"name": obj_name, "coords": coords, "order": [], "tris": []})
        return parts[-1]

    with open(filename) as f:
        for lineno, line in enumerate(f, 1):
            tok = line.split()
            if not tok:
                continue
            key = tok[0].upper()

            if in_header:
                if key == "}":
                    in_header = False
                elif ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().lstrip("=").strip()
                    if k.strip().lower() == "name" and v:
                        # the HEADER precedes the TFACEs of its object, so it
                        # names the parts still to come, not the previous one
                        obj_name = v
                continue

            if key == "GOCAD":
                obj_name = default_name
                coords = {}
                part = None
                continue
            if key == "HEADER":
                in_header = "}" not in line
                continue
            if key == "TFACE":
                part = new_part()
                continue
            if key == "END":
                part = None
                continue

            if key in ("VRTX", "PVRTX", "ATOM", "PATOM"):
                gid = int(tok[1])
                if key in ("VRTX", "PVRTX"):
                    coords[gid] = (float(tok[2]), float(tok[3]), float(tok[4]))
                else:
                    # an ATOM is a vertex sharing the position of another one
                    coords[gid] = coords[int(tok[2])]
                if part is None:  # geometry before any TFACE
                    part = new_part()
                part["order"].append(gid)
            elif key == "TRGL":
                if len(tok) < 4:
                    raise ValueError(f"{filename}:{lineno}: malformed TRGL")
                if part is None:
                    part = new_part()
                part["tris"].append(tuple(int(t) for t in tok[1:4]))

    result = []
    for p in parts:
        if not p["tris"]:
            continue
        # Gocad keeps unused vertices around, and a TRGL may reference a vertex
        # declared under another TFACE — so index the referenced ones only,
        # declaration order first, then any foreign vertex as it appears.
        pcoords = p["coords"]
        used = {g for tri in p["tris"] for g in tri}
        local = {}
        for gid in p["order"]:
            if gid in used and gid not in local:
                local[gid] = len(local)
        for tri in p["tris"]:
            for gid in tri:
                if gid not in local:
                    if gid not in pcoords:
                        raise ValueError(
                            f"{filename}: triangle uses unknown vertex {gid}"
                        )
                    local[gid] = len(local)

        positions = [0.0] * (3 * len(local))
        for gid, l in local.items():
            positions[3 * l : 3 * l + 3] = pcoords[gid]
        indices = [local[g] for tri in p["tris"] for g in tri]
        result.append({"name": p["name"], "positions": positions, "indices": indices})

    if not result:
        raise ValueError(f"{filename}: no triangulated surface found")
    return result


# ============================================================================
# 0bis. Gocad TSurf (.ts) writer
# ============================================================================
def save_gocad_ts(filename, surfaces, names=None):
    """Write `surfaces` as a Gocad TSurf file, one TSurf object per surface.

    Each vertex carries the slip (Burgers vector) in the element-local frame,
    averaged from the triangles around it, as the properties
    `slip_normal slip_strike slip_dip slip_mag` (metres).
    """

    # arch's element-local frame is ordered (normal, strike, dip) — Triangle.cxx
    # builds l2g_ = Matrix33(normal, strike, dip), which is what the string bc API
    # ("normal" -> 0, "strike" -> 1, "dip" -> 2) uses. Keep that order here so the
    # property names match the components they carry.
    LOCAL_AXES = ("normal", "strike", "dip")

    def vertex_average(surface, values, ncomp):
        """Spread a per-triangle quantity onto the vertices of `surface`.

        `values` is flat, `ncomp` components per triangle; the result is flat with
        `ncomp` components per vertex, each the mean over the incident triangles.
        """
        indices = surface.indices()
        out = [0.0] * (surface.nbVertices() * ncomp)
        count = [0] * surface.nbVertices()

        for t in range(surface.nbTriangles()):
            for k in range(3):
                v = indices[3 * t + k]
                count[v] += 1
                for c in range(ncomp):
                    out[ncomp * v + c] += values[ncomp * t + c]

        for v, n in enumerate(count):
            if n:
                for c in range(ncomp):
                    out[ncomp * v + c] /= n
        return out

    with open(filename, "w") as o:
        for i, surface in enumerate(surfaces):
            name = names[i] if names else f"surface_{i}"
            vertices = surface.vertices()
            indices = surface.indices()

            # local=True (element frame), atTriangles=True (one value per element)
            slip = vertex_average(surface, surface.displ(True, True), 3)

            o.write("GOCAD TSurf 1.0\n")
            o.write("HEADER {\nname:%s\n}\n" % name)
            o.write("PROPERTIES %s slip_mag\n" % " ".join("slip_" + a for a in LOCAL_AXES))
            o.write("PROPERTY_CLASSES %s slip_mag\n" % " ".join("slip_" + a for a in LOCAL_AXES))
            o.write("NO_DATA_VALUES %s\n" % " ".join(["-99999"] * 4))
            o.write("TFACE\n")
            for v in range(surface.nbVertices()):
                x, y, z = vertices[3 * v : 3 * v + 3]
                sn, ss, sd = slip[3 * v : 3 * v + 3]
                mag = (sn * sn + ss * ss + sd * sd) ** 0.5
                # %.12g on the coordinates: %g's 6 significant digits would
                # round a 9 km deep vertex to the centimetre
                o.write(
                    "PVRTX %d %.12g %.12g %.12g %.9g %.9g %.9g %.9g\n"
                    % (v, x, y, z, sn, ss, sd, mag)
                )
            for t in range(surface.nbTriangles()):
                o.write("TRGL %d %d %d\n" % tuple(indices[3 * t : 3 * t + 3]))
            o.write("END\n")


# ============================================================================
# 1. Model + material
# ============================================================================
model = arch.Model()
model.setHalfSpace(True)
model.setMaterial(0.22, 4.8e10, 2700)

# ============================================================================
# 2. Fault surface (the discontinuity)
# ============================================================================
ts_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "EW-DL.ts")
parts = load_gocad_ts(ts_file)
print(f"Loaded {ts_file}: {len(parts)} TFACE(s)")

faults = []
fault_names = []
for i, p in enumerate(parts):
    fault = model.addSurface(p["positions"], p["indices"])

    # Boundary conditions per local component
    fault.setBcType("dip", "free")     # allow dip-slip
    fault.setBcType("strike", "free")  # allow strike-slip
    fault.setBcType("normal", "free")  # allow fault opening/closing
    fault.setBcValue("normal", lambda x, y, z: 0)

    c = arch.Coulomb()
    c.setFriction(0.1)
    fault.addTic(c)

    faults.append(fault)
    # the TFACEs of a single TSurf share one name — suffix it so the objects
    # stay distinguishable once written back out
    fault_names.append(p["name"] if len(parts) == 1 else f"{p['name']}_{i}")
    print(
        f"  [{i}] {fault_names[-1]}: {fault.nbTriangles()} triangles, "
        f"{fault.nbVertices()} vertices"
    )

print(f"Model dof: {model.nbDof()}")

# ============================================================================
# 3. Far-field (remote) stress
# ============================================================================
remote = arch.RemoteStress()
remote.seth(-76500000)
remote.setH(-90000000)
remote.setv(-45000000)
remote.setTheta(135)
model.addRemote(remote)


# ============================================================================
# 4. Solve (Gauss-Seidel iterative BEM solver)
# ============================================================================
# solver = arch.SeidelSolver(model)
solver = arch.GmresSolver(model)
solver.setCgnr()
solver.setEps(1e-9)
solver.setMaxIter(200)
ok = solver.run(True)


# ============================================================================
# 5. Save the faults, with their slip, as a Gocad TSurf file
# ============================================================================
out_file = os.path.splitext(ts_file)[0] + "_result.ts"
save_gocad_ts(out_file, faults, fault_names)
print(f"\nResult written to {out_file}")

print("\n--- Slip on the faults (element-local frame, m) ---")
print(f"  {'surface':<12} {'max |slip|':>12} {'mean |slip|':>12}")
for name, fault in zip(fault_names, faults):
    d = fault.displ(True, True)  # local=True, atTriangles=True
    mag = [
        (d[i] ** 2 + d[i + 1] ** 2 + d[i + 2] ** 2) ** 0.5
        for i in range(0, len(d), 3)
    ]
    print(f"  {name:<12} {max(mag):12.4e} {sum(mag) / len(mag):12.4e}")
