#!/usr/bin/env python3
"""Headless renderer for the NVS benchmark (Blender as a Python module, bpy 4.x).

    python scripts/nvs/render_bpy.py --list data/nvs/objaverse_index.json --out ~/data/nvs/objaverse \
        --protocol train --n-views 32 --engine cycles --samples 32 --shard 0 --nshards 3
    python scripts/nvs/render_bpy.py --list data/nvs/gso_index.json --out ~/data/nvs/gso --protocol gso

Protocol (smr.nvs.cameras): object normalised to a bounding sphere of radius 0.5 at the
origin, cameras on a sphere of radius 2.0, vertical FOV 40 deg, square `--res` images,
white world (lighting) plus one sun (direction fixed per object), Cycles GPU (OptiX/CUDA)
with denoising, "Standard" view transform (no filmic tone curve), PNG RGBA with a transparent film:
alpha is the object mask, and loaders composite the colour onto white (LVSM/GS-LRM convention).
Per object: <out>/<id>/{000.png,...} and cams.json (OpenCV c2w, y-up world) + meta.json.
Resume-safe: objects whose cams.json exists are skipped.  Batches of --batch objects share
one Blender session (read_factory_settings between them) to amortise start-up.

VERIFY-ON-SERVER: bpy API names used here follow Blender 4.x (obj_import, gltf import,
cycles preferences).  Run the dry run (UPDATE_NOTES_v139.md) and LOOK at one render of a
GSO object: the 'input' views must show the object from 20 deg above.
"""
import argparse, json, math, pathlib, sys, time, zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from smr.nvs import cameras as C  # noqa: E402

# y-up (ours / glTF) -> z-up (Blender): x->x, y->z, z->-y
M_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
# OpenCV camera (z forward, y down) -> Blender camera (-z forward, y up)
CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def setup_cycles_prefs(bpy, device_type):
    """GPU devices for Cycles (preferences persist across factory resets)."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device_type
    prefs.get_devices()
    n_gpu = 0
    for d in prefs.devices:
        d.use = d.type != "CPU"
        n_gpu += int(d.type != "CPU")
    return n_gpu


def apply_cycles_scene(bpy, samples, n_gpu):
    """Per-scene render settings (must be re-applied after read_factory_settings)."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if n_gpu else "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = 4
    scene.render.film_transparent = True                      # background alpha 0; world still lights the object
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


def setup_scene(bpy, res):
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"          # alpha = object mask; loaders composite on white
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    # white uniform world (background + ambient)
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    scene.world = world
    # camera
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(C.FOV_DEG)
    cam_data.clip_start, cam_data.clip_end = 0.05, 100.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def add_sun(bpy, rng):
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 2.0
    sun_data.angle = math.radians(10)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    az, el = rng.uniform(0, 2 * math.pi), rng.uniform(math.radians(25), math.radians(70))
    d = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])  # y-up direction TO the light
    # a sun points along its local -Z; aim it from `d` toward the origin (in Blender z-up coords)
    d_b = (M_YUP_TO_ZUP[:3, :3] @ d)
    import mathutils
    sun.rotation_euler = mathutils.Vector(tuple(-d_b)).to_track_quat("-Z", "Y").to_euler()
    return sun


def relink_missing_images(bpy, root):
    """Textures referenced by a bare filename (GSO's MTL) live elsewhere in the model folder:
    point every image whose file is missing at a same-named file found under `root`."""
    root = pathlib.Path(root); n = 0
    for img in bpy.data.images:
        fp = bpy.path.abspath(img.filepath) if img.filepath else ""
        if not fp or pathlib.Path(fp).exists():
            continue
        cands = list(root.glob(f"**/{pathlib.Path(fp).name}"))
        if cands:
            img.filepath = str(cands[0]); img.reload(); n += 1
    return n


def import_object(bpy, path, obj_up):
    path = str(path)
    before = set(bpy.data.objects)
    if path.lower().endswith((".glb", ".gltf")):
        try:
            bpy.ops.import_scene.gltf(filepath=path, loglevel=30)     # WARNING: no per-node INFO spam
        except TypeError:
            bpy.ops.import_scene.gltf(filepath=path)
    elif path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path, up_axis=obj_up, forward_axis="NEGATIVE_Z" if obj_up == "Y" else "Y")
    else:
        raise ValueError(f"unsupported mesh format: {path}")
    # model root = the directory holding the mesh's parent (meshes/..) for GSO, the file's folder otherwise
    model_root = pathlib.Path(path).parent.parent if pathlib.Path(path).parent.name == "meshes" else pathlib.Path(path).parent
    n_fixed = relink_missing_images(bpy, model_root)
    return [o for o in bpy.data.objects if o not in before], n_fixed


def normalise(bpy, objs):
    """Parent everything to an empty and set its location/scale so the mesh vertices lie in a
    sphere of radius 0.5 about the origin (bounding-box centre at the origin)."""
    import mathutils
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh objects imported")
    pts = []
    for o in meshes:
        mw = np.array(o.matrix_world)
        n = len(o.data.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        o.data.vertices.foreach_get("co", co)         # fast bulk read
        co = co.reshape(-1, 3).astype(np.float64)
        pts.append(co @ mw[:3, :3].T + mw[:3, 3])
    P = np.concatenate(pts)
    lo, hi = P.min(0), P.max(0)
    centre = 0.5 * (lo + hi)
    radius = np.linalg.norm(P - centre, axis=1).max()
    if radius < 1e-9:
        raise RuntimeError("degenerate object (zero extent)")
    scale = 0.5 / radius
    root = bpy.data.objects.new("norm_root", None)
    bpy.context.scene.collection.objects.link(root)
    for o in objs:
        if o.parent is None:
            o.parent = root
    root.scale = (scale, scale, scale)
    root.location = tuple(-scale * centre)
    bpy.context.view_layer.update()
    return dict(centre=centre.tolist(), radius=float(radius), scale=float(scale),
                n_vertices=int(sum(len(o.data.vertices) for o in meshes)))


def set_camera(cam, c2w_cv_yup):
    import mathutils
    m = M_YUP_TO_ZUP @ np.asarray(c2w_cv_yup) @ CV_TO_BLENDER_CAM
    cam.matrix_world = mathutils.Matrix(m.tolist())


def render_object(bpy, entry, out_dir, protocol, n_views, res, seed, obj_up, log, samples, n_gpu):
    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    apply_cycles_scene(bpy, samples, n_gpu)
    cam = setup_scene(bpy, res)
    rng = np.random.default_rng(seed)
    add_sun(bpy, rng)
    t0 = time.time()
    objs, n_relinked = import_object(bpy, entry["path"], obj_up)
    norm = normalise(bpy, objs)
    t_import = time.time() - t0
    missing = [i.name for i in bpy.data.images if i.filepath and not pathlib.Path(bpy.path.abspath(i.filepath)).exists()]
    views = C.gso_views(seed) if protocol == "gso" else C.train_views(n_views, seed)
    K = C.intrinsics(res)
    times = []
    for i, v in enumerate(views):
        set_camera(cam, v["c2w"])
        bpy.context.scene.render.filepath = str(out_dir / f"{i:03d}.png")
        t = time.time(); bpy.ops.render.render(write_still=True); times.append(time.time() - t)
    json.dump(C.cams_to_json(views, K, res), open(out_dir / "cams.json", "w"))
    json.dump(dict(id=entry["id"], source=entry["path"], protocol=protocol, seed=seed, normalisation=norm,
                   render_seconds_per_view=float(np.mean(times)), import_seconds=float(t_import),
                   textures_relinked=int(n_relinked), textures_missing=missing),
              open(out_dir / "meta.json", "w"), indent=1)
    log(f"{entry['id']}: {len(views)} views, {np.mean(times):.2f} s/view, {norm['n_vertices']:,} verts"
        + (f", {n_relinked} texture(s) relinked" if n_relinked else "") + (f", MISSING textures: {missing}" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="json list of {id, path} entries (objaverse_index.json / gso_index.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["train", "gso"], default="train")
    ap.add_argument("--n-views", type=int, default=32)
    ap.add_argument("--res", type=int, default=C.RES)
    ap.add_argument("--engine", choices=["cycles"], default="cycles")
    ap.add_argument("--device", choices=["OPTIX", "CUDA"], default="OPTIX")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--obj-up", choices=["Y", "Z"], default="Z", help="up axis of .obj inputs (GSO meshes are z-up)")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N entries of the shard (dry runs)")
    ap.add_argument("--batch", type=int, default=50, help="objects per Blender session (state reset between objects)")
    a = ap.parse_args()

    import bpy  # noqa: E402
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("glTFImporter", "io_scene_gltf2", "io.import.gltf2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    entries = json.load(open(a.list))
    entries = [e for i, e in enumerate(entries) if i % a.nshards == a.shard]
    if a.limit:
        entries = entries[: a.limit]
    out = pathlib.Path(a.out).expanduser()
    log_path = out / f"render_shard{a.shard}.log"; out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); open(log_path, "a").write(line + "\n")

    n_gpu = setup_cycles_prefs(bpy, a.device)
    log(f"cycles: {n_gpu} GPU device(s) via {a.device}; {len(entries)} objects in shard {a.shard}/{a.nshards}; protocol {a.protocol}")
    done = failed = 0
    for e in entries:
        od = out / e["id"]
        if (od / "cams.json").exists():
            continue
        try:
            seed = int(e.get("seed", zlib.crc32(e["id"].encode()) % (2 ** 31)))   # deterministic across processes
            render_object(bpy, e, od, a.protocol, a.n_views, a.res, seed, a.obj_up, log, a.samples, n_gpu)
            done += 1
        except Exception as ex:                    # one bad asset must not stop the shard
            failed += 1
            log(f"{e['id']}: FAILED {type(ex).__name__}: {ex}")
            (od).mkdir(parents=True, exist_ok=True); (od / "FAILED").write_text(str(ex))
    log(f"shard {a.shard}: rendered {done}, failed {failed}")


if __name__ == "__main__":
    main()
