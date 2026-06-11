# convert STEP/STL/OBJ meshes to the training format

import os
import csv
import json
import hashlib
import logging
import argparse

from meshio import load_stl, save_obj

logger = logging.getLogger(__name__)

STEP_EXT = {".step", ".stp"}
MESH_EXT = {".stl"}
CONVERTIBLE_EXT = STEP_EXT | MESH_EXT

# categories filtered out (not connectors)
IGNORED_CATEGORIES = {
    "cabinet engineering", "not categorized", "fluid engineering",
    "electrical installation", "cables and conductors",
    "measuring+reporting devices", "knx (eib)",
    "power source", "loads", "accessories",
}

# STEP -> STL  (needs a CAD kernel)

def step_to_stl(step_path, stl_path, linear_deflection=0.1, angular_deflection=0.5):
    """Tessellate STEP -> STL. Tries gmsh then OpenCASCADE."""
    backend = _pick_step_backend()
    if backend == "gmsh":
        _step_gmsh(step_path, stl_path, linear_deflection)
    elif backend == "occ":
        _step_occ(step_path, stl_path, linear_deflection, angular_deflection)
    else:
        raise RuntimeError(
            "No CAD kernel found. Install gmsh or cadquery-ocp:\n"
            "  pip install gmsh\n  pip install cadquery-ocp")
    return stl_path

def _pick_step_backend():
    import importlib.util
    if importlib.util.find_spec("gmsh"):
        return "gmsh"
    for mod in ("OCC", "OCP"):
        if importlib.util.find_spec(mod):
            return "occ"
    return None

def _step_gmsh(step_path, stl_path, linear_deflection):
    import gmsh  # type: ignore[import]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", linear_deflection)
        gmsh.open(step_path)
        gmsh.model.mesh.generate(2)
        gmsh.write(stl_path)
    finally:
        gmsh.finalize()

def _step_occ(step_path, stl_path, linear_deflection, angular_deflection):
    try:
        from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore[import]
        from OCC.Core.StlAPI import StlAPI_Writer  # type: ignore[import]
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore[import]
        from OCC.Core.IFSelect import IFSelect_RetDone  # type: ignore[import]
    except ImportError:
        from OCP.STEPControl import STEPControl_Reader  # type: ignore[import]
        from OCP.StlAPI import StlAPI_Writer  # type: ignore[import]
        from OCP.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore[import]
        from OCP.IFSelect import IFSelect_RetDone  # type: ignore[import]
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    StlAPI_Writer().Write(shape, stl_path)

# STL -> OBJ  (pure numpy, via meshio)

def stl_to_obj(stl_path, obj_path):
    """Read STL, weld vertices into an indexed mesh, and save as OBJ."""
    V, F = load_stl(stl_path)
    save_obj(obj_path, V, F)
    logger.debug("STL->OBJ: %s -> %s  (%d v, %d f)", stl_path, obj_path, len(V), len(F))
    return obj_path

# convert a single file to OBJ (STEP via STL, or STL directly)

def convert_file(src_path, out_dir, keep_stl=False, linear_deflection=0.1):
    """Convert STEP/STL -> OBJ. Returns OBJ path."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    ext = os.path.splitext(src_path)[1].lower()
    obj_path = os.path.join(out_dir, stem + ".obj")

    if ext in STEP_EXT:
        stl_path = os.path.join(out_dir, stem + ".stl")
        step_to_stl(src_path, stl_path, linear_deflection=linear_deflection)
        stl_to_obj(stl_path, obj_path)
        if not keep_stl:
            try:
                os.remove(stl_path)
            except OSError:
                pass
    elif ext in MESH_EXT:
        stl_to_obj(src_path, obj_path)
    else:
        raise ValueError(f"Unsupported: {ext} (only {sorted(CONVERTIBLE_EXT)})")
    return obj_path

# conversion with time limit (300s per file)

# Some very large or degenerate STEP files can keep the CAD kernel busy
# indefinitely. Running conversion in a subprocess and killing it after the
# timeout means one bad file cannot stall the entire run.

def _convert_worker(src, out_dir, keep_stl, linear_deflection, q):
    try:
        obj = convert_file(src, out_dir, keep_stl=keep_stl,
                           linear_deflection=linear_deflection)
        q.put(("ok", obj))
    except Exception as exc:
        q.put(("err", str(exc)))

def convert_file_timed(src_path, out_dir, keep_stl=False, linear_deflection=0.1,
                       timeout_sec=None):
    """Like convert_file, but aborts after timeout_sec.

    timeout_sec None or 0 -> no limit (identical to convert_file). Otherwise
    the conversion runs in a subprocess that is terminated after the timeout
    (raises TimeoutError).
    """
    if not timeout_sec:
        return convert_file(src_path, out_dir, keep_stl=keep_stl,
                            linear_deflection=linear_deflection)

    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    q = mp.Queue()
    p = mp.Process(target=_convert_worker,
                   args=(src_path, out_dir, keep_stl, linear_deflection, q))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Conversion exceeded {timeout_sec:.0f}s: {src_path}")
    try:
        status, payload = q.get(timeout=10)
    except Exception:
        raise RuntimeError(f"Conversion returned no result: {src_path}")
    if status == "err":
        raise RuntimeError(payload)
    return payload

# geometric dedup check via SHA-256

# Many parts differ only in electrical specs but have identical geometry
# (54.12% of STEP files were duplicates). SHA-256 hashes the files to find them.

def sha256_file(path, chunk=1 << 20):
    """SHA-256 hash of a file (streaming, memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def find_geometric_duplicates(root, exts=(".stl", ".obj", ".off")):
    """Group files under 'root' by content hash.

    Returns dict {hash -> [paths]}. Groups with >1 entry are duplicates.
    """
    exts = {e.lower() for e in exts}
    by_hash = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            fp = os.path.join(dirpath, name)
            try:
                by_hash.setdefault(sha256_file(fp), []).append(fp)
            except OSError:
                continue
    return by_hash

def dedupe_tree(root, exts=(".stl", ".obj", ".off"), apply=False):
    """Find and optionally delete geometric duplicates (keep first per hash).

    apply=False -> dry run (list only).
    apply=True  -> delete all but the first file per hash group.
    Returns dict with n_total, n_unique, n_duplicates, removed.
    """
    by_hash = find_geometric_duplicates(root, exts)
    n_total = sum(len(v) for v in by_hash.values())
    n_unique = len(by_hash)
    removed = []
    for paths in by_hash.values():
        for extra in sorted(paths)[1:]:
            if apply:
                try:
                    os.remove(extra)
                    removed.append(extra)
                except OSError:
                    pass
            else:
                removed.append(extra)
    dup_rate = (n_total - n_unique) / n_total * 100 if n_total else 0.0
    logger.info("dedupe_tree: %d files, %d unique, %d duplicates (%.1f%%)%s",
                n_total, n_unique, n_total - n_unique, dup_rate,
                " -> deleted" if apply else " (dry run)")
    return {"n_total": n_total, "n_unique": n_unique,
            "n_duplicates": n_total - n_unique, "removed": removed}

# category filter

def filter_by_category(items, metadata, ignored=None, key="category"):
    """Filter items by their category metadata.

    items    : iterable of ids/filenames.
    metadata : dict {id -> {category, subcategory, technology, ...}}.
    ignored  : set of category names to discard (default: IGNORED_CATEGORIES).
    Returns  : list of kept ids (category not in ignored).
    """
    ignored = {c.lower() for c in (ignored or IGNORED_CATEGORIES)}
    kept = []
    for it in items:
        meta = metadata.get(it, {})
        cats = {str(meta.get(k, "")).lower()
                for k in (key, "subcategory", "technology")}
        if cats & ignored:
            continue
        kept.append(it)
    return kept

# dataset statistics

def dataset_stats(root, manufacturer_depth=1):
    """Collect statistics over a dataset directory tree.

    manufacturer_depth: which directory level below 'root' holds the
    manufacturer name (default 1 = direct subfolders of root).

    Returns dict with n_files, total_bytes, by_ext, by_manufacturer.
    """
    n_files = 0
    total_bytes = 0
    by_ext = {}
    by_manu = {}

    root = os.path.abspath(root)
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        manu = parts[manufacturer_depth - 1] if len(parts) >= manufacturer_depth else "(root)"
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in (CONVERTIBLE_EXT | {".obj"}):
                continue
            fp = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(fp)
            except OSError:
                continue
            n_files += 1
            total_bytes += size
            by_ext[ext] = by_ext.get(ext, 0) + 1
            by_manu[manu] = by_manu.get(manu, 0) + 1

    return {
        "n_files": n_files,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024 ** 3), 2),
        "n_manufacturers": len([m for m in by_manu if m != "(root)"]) or len(by_manu),
        "by_ext": dict(sorted(by_ext.items())),
        "by_manufacturer": dict(sorted(by_manu.items(), key=lambda kv: -kv[1])),
    }

def print_stats(stats):
    print("Dataset statistics")
    print(f"  Total files        : {stats['n_files']:,}")
    print(f"  Total size         : {stats['total_gb']} GB")
    print(f"  Manufacturers      : {stats['n_manufacturers']}")
    print(f"  Formats            : {stats['by_ext']}")
    top = list(stats["by_manufacturer"].items())[:10]
    if top:
        print("  Top manufacturers (by file count):")
        for manu, cnt in top:
            print(f"     {manu:30s} {cnt:,}")

# convert entire tree + write manifest

def convert_tree(root, out_dir, manifest_path=None, keep_stl=False,
                 linear_deflection=0.1, flatten=True, timeout_sec=300):
    """Convert all STEP/STL files under 'root' to OBJ.

    flatten=True puts all OBJs flat in out_dir; if names clash the relative
    path is encoded into the filename to avoid overwrites.
    flatten=False mirrors the folder structure under out_dir.

    timeout_sec limits conversion time per file (default 300s);
    0/None disables the limit. A timed-out file is logged as an error and
    does not stop the run.

    Writes a manifest CSV with source, destination, status per file.
    Returns dict with counts {ok, errors, manifest}.
    """
    root = os.path.abspath(root)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    ok = err = 0

    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in CONVERTIBLE_EXT:
                continue
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, root)

            if flatten:
                stem = os.path.splitext(rel)[0].replace(os.sep, "__")
                target_dir = out_dir
                obj_name = stem + ".obj"
            else:
                target_dir = os.path.join(out_dir, os.path.dirname(rel))
                obj_name = os.path.splitext(name)[0] + ".obj"

            try:
                os.makedirs(target_dir, exist_ok=True)
                obj_path = convert_file_timed(src, target_dir, keep_stl=keep_stl,
                                              linear_deflection=linear_deflection,
                                              timeout_sec=timeout_sec)
                final = os.path.join(target_dir, obj_name)
                if os.path.abspath(obj_path) != os.path.abspath(final):
                    os.replace(obj_path, final)
                    obj_path = final
                rows.append({"src": rel, "obj": os.path.relpath(obj_path, out_dir), "status": "ok"})
                ok += 1
            except Exception as exc:
                logger.warning("Conversion failed: %s (%s)", rel, exc)
                rows.append({"src": rel, "obj": "", "status": f"error: {exc}"})
                err += 1

    if manifest_path is None:
        manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["src", "obj", "status"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("convert_tree done: %d ok, %d errors -> manifest %s", ok, err, manifest_path)
    return {"ok": ok, "errors": err, "manifest": manifest_path}

# CLI

def _build_parser():
    p = argparse.ArgumentParser(
        description="CAD dataset prep: STEP->STL->OBJ + stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python dataset_convert.py stats   ./data
  python dataset_convert.py dedupe  ./data --apply
  python dataset_convert.py file    part.step --out ./obj
  python dataset_convert.py tree    ./data --out ./obj
""",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("stats", help="dataset statistics")
    ps.add_argument("root")
    ps.add_argument("--json")

    pd = sub.add_parser("dedupe", help="find/remove geometric duplicates")
    pd.add_argument("root")
    pd.add_argument("--apply", action="store_true")
    pd.add_argument("--ext", default=".stl,.obj,.off")

    pf = sub.add_parser("file", help="convert single file")
    pf.add_argument("src")
    pf.add_argument("--out", default="./obj")
    pf.add_argument("--deflection", type=float, default=0.1)
    pf.add_argument("--keep-stl", action="store_true")

    pt = sub.add_parser("tree", help="convert entire tree + manifest")
    pt.add_argument("root")
    pt.add_argument("--out", default="./obj")
    pt.add_argument("--manifest")
    pt.add_argument("--deflection", type=float, default=0.1)
    pt.add_argument("--keep-stl", action="store_true")
    pt.add_argument("--mirror", action="store_true")
    pt.add_argument("--timeout", type=float, default=300)
    return p

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _build_parser().parse_args()

    if args.cmd == "stats":
        stats = dataset_stats(args.root)
        print_stats(stats)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2, ensure_ascii=False)
            logger.info("Stats saved: %s", args.json)

    elif args.cmd == "dedupe":
        exts = tuple(e if e.startswith(".") else "." + e for e in args.ext.split(","))
        res = dedupe_tree(args.root, exts=exts, apply=args.apply)
        logger.info("Result: %d/%d unique, %d duplicates",
                    res["n_unique"], res["n_total"], res["n_duplicates"])

    elif args.cmd == "file":
        obj = convert_file(args.src, args.out, keep_stl=args.keep_stl,
                           linear_deflection=args.deflection)
        logger.info("Done: %s", obj)

    elif args.cmd == "tree":
        res = convert_tree(args.root, args.out, manifest_path=args.manifest,
                           keep_stl=args.keep_stl, linear_deflection=args.deflection,
                           flatten=not args.mirror, timeout_sec=args.timeout)
        logger.info("Result: %s", res)

if __name__ == "__main__":
    main()
