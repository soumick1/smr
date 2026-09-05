"""Renderer geometry test with a mock `bpy`/`mathutils` (no Blender needed).

Checks the two coordinate conversions (y-up world -> Blender z-up; OpenCV camera -> Blender
camera): the Blender camera's -Z axis must point at the origin, its +Y axis must be upright
(positive z in Blender for elevated views); normalise() must bring every vertex within a
sphere of radius 0.5 through the parent empty's location/scale; and the object loop must
skip finished objects, write cams.json/meta.json, and mark a failing asset without aborting.
"""
import json, pathlib, sys, tempfile, types
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "nvs"))


# ----------------------------------------------------------------- mock mathutils / bpy ---
class Vector(tuple):
    def __new__(cls, xyz): return super().__new__(cls, tuple(float(v) for v in xyz))
    def to_track_quat(self, track, up):
        assert track == "-Z" and up == "Y"
        return _Quat(np.array(self))
    def __neg__(self): return Vector([-v for v in self])


class _Quat:
    def __init__(self, dirn): self.d = dirn / np.linalg.norm(dirn)
    def to_euler(self): return ("euler", tuple(self.d))          # the sun test reads the direction back


class Matrix:
    def __init__(self, rows): self.a = np.array(rows, float)
    def __matmul__(self, other):
        if isinstance(other, Vector):
            v = self.a @ np.array(list(other) + [1.0]); return Vector(v[:3])
        return Matrix(self.a @ other.a)
    def __array__(self, dtype=None, copy=None): return self.a.astype(dtype) if dtype else self.a


class _Vert:
    def __init__(self, co): self.co = Vector(co)


class _Verts(list):
    def foreach_get(self, attr, out):
        assert attr == "co"
        out[:] = np.array([v.co for v in self]).ravel()


class _Mesh:
    def __init__(self, verts): self.vertices = _Verts([_Vert(v) for v in verts])


class _Obj:
    def __init__(self, name, typ, data=None):
        self.name, self.type, self.data, self.parent = name, typ, data, None
        self.matrix_world = Matrix(np.eye(4)); self.scale = (1, 1, 1); self.location = (0, 0, 0)
        self.rotation_euler = None


class _Objects(list):
    def new(self, name, data):
        typ = "MESH" if isinstance(data, _Mesh) else ("CAMERA" if isinstance(data, _CamData) else ("LIGHT" if isinstance(data, _Light) else "EMPTY"))
        return _Obj(name, typ, data)


class _CamData:
    def __init__(self): self.sensor_fit = None; self.angle_y = None; self.clip_start = None; self.clip_end = None


class _Light:
    def __init__(self): self.energy = None; self.angle = None


class _NS(types.SimpleNamespace):
    pass


class MockBpy:
    """Just enough of bpy for scripts/nvs/render_bpy.py."""

    def __init__(self, fail_id=None):
        self.fail_id = fail_id; self.rendered = []
        self._reset()
        cyc = _NS(devices=[_NS(type="CUDA", use=False), _NS(type="CPU", use=False)], compute_device_type=None, get_devices=lambda: None)
        self.context = _NS(scene=None, preferences=_NS(addons={"cycles": _NS(preferences=cyc)}),
                           view_layer=_NS(update=lambda: None))
        self.context.scene = self.scene
        self.data = _NS(objects=self.objects, worlds=_NS(new=self._new_world), cameras=_NS(new=lambda n: _CamData()),
                        lights=_NS(new=lambda n, type=None: _Light()), images=[])
        self.path = _NS(abspath=lambda p: p)
        self.ops = _NS(wm=_NS(read_factory_settings=self._factory, obj_import=self._import_obj),
                       import_scene=_NS(gltf=self._import_glb), render=_NS(render=self._render))

    def _reset(self):
        self.objects = _Objects()
        coll = _NS(objects=_NS(link=lambda o: self.objects.append(o)))
        self.scene = _NS(render=_NS(engine=None, resolution_x=0, resolution_y=0, resolution_percentage=0, filepath="",
                                    film_transparent=None, image_settings=_NS(file_format=None, color_mode=None, color_depth=None)),
                         cycles=_NS(), view_settings=_NS(), world=None, camera=None, collection=coll)

    def _factory(self, use_empty=False):
        self._reset(); self.context.scene = self.scene; self.data.objects = self.objects

    def _new_world(self, name):
        bg = _NS(inputs=[_NS(default_value=None), _NS(default_value=None)])
        return _NS(use_nodes=False, node_tree=_NS(nodes=_NS(get=lambda n: bg)))

    def _import_glb(self, filepath, loglevel=None):
        if self.fail_id and self.fail_id in filepath:
            raise RuntimeError("corrupt glb")
        # a 2x3x1 box offset from the origin, plus a huge world matrix on a second part
        box = np.array([[x, y, z] for x in (0, 2) for y in (0, 3) for z in (0, 1)], float) + np.array([10, -4, 7])
        o1 = _Obj("part1", "MESH", _Mesh(box)); o2 = _Obj("part2", "MESH", _Mesh(box * 0.5))
        o2.matrix_world = Matrix(np.diag([2.0, 2.0, 2.0, 1.0]))
        self.objects += [o1, o2]

    def _import_obj(self, filepath, up_axis=None, forward_axis=None):
        self._import_glb(filepath)

    def _render(self, write_still=False):
        pathlib.Path(self.scene.render.filepath).write_bytes(b"png"); self.rendered.append(self.scene.render.filepath)


def with_mock(fail_id=None):
    import mathutils_mock  # noqa: F401  (installed below)
    bpy = MockBpy(fail_id)
    return bpy


# install the mock mathutils before importing the renderer
sys.modules["mathutils"] = types.SimpleNamespace(Vector=Vector, Matrix=Matrix)
sys.modules["mathutils_mock"] = sys.modules["mathutils"]
import render_bpy as R  # noqa: E402
from smr.nvs import cameras as C  # noqa: E402


def test_camera_conversion_looks_at_origin_upright():
    for v in C.gso_views(seed=0) + C.train_views(10, seed=2):
        cam = _Obj("cam", "CAMERA", _CamData())
        R.set_camera(cam, v["c2w"])
        Mb = np.array(cam.matrix_world)
        pos, fwd, up = Mb[:3, 3], -Mb[:3, 2], Mb[:3, 1]           # Blender camera looks along -Z, +Y up
        assert np.allclose(np.linalg.norm(pos), C.RADIUS)
        assert np.dot(fwd, -pos / np.linalg.norm(pos)) > 0.9999    # points at the origin
        if v["el"] > -10:
            assert up[2] > 0                                        # upright in z-up Blender space
        assert abs(pos[2] - C.RADIUS * np.sin(np.deg2rad(v["el"]))) < 1e-9   # elevation maps to Blender z


def test_normalise_and_loop():
    bpy = with_mock(fail_id="bad")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        lst = tmp / "list.json"; json.dump([dict(id="good1", path="/x/good1.glb"), dict(id="bad", path="/x/bad.glb")], open(lst, "w"))
        (tmp / "out" / "skip_me").mkdir(parents=True); (tmp / "out" / "skip_me" / "cams.json").write_text("{}")
        entries = json.load(open(lst)) + [dict(id="skip_me", path="/x/skip.glb")]
        done = failed = skipped = 0
        for e in entries:
            od = tmp / "out" / e["id"]
            if (od / "cams.json").exists():
                skipped += 1; continue
            try:
                R.render_object(bpy, e, od, "gso", 32, 64, 7, "Z", lambda m: None, samples=8, n_gpu=1); done += 1
                # normalisation: apply root transform to every vertex -> within radius 0.5
                root = [o for o in bpy.objects if o.name == "norm_root"][0]
                s = root.scale[0]; t = np.array(root.location)
                for o in bpy.objects:
                    if o.type == "MESH":
                        co = np.array([v.co for v in o.data.vertices]); mw = np.array(o.matrix_world)
                        world = co @ mw[:3, :3].T + mw[:3, 3]
                        assert (np.linalg.norm(s * world + t, axis=1) <= 0.5 + 1e-9).all()
                meta = json.load(open(od / "meta.json")); cams = json.load(open(od / "cams.json"))
                assert len(cams["views"]) == 14 and meta["normalisation"]["n_vertices"] == 16 and len(list(od.glob("*.png"))) == 14
            except Exception:
                failed += 1; od.mkdir(parents=True, exist_ok=True); (od / "FAILED").write_text("x")
        assert (done, failed, skipped) == (1, 1, 1)
        assert bpy.scene.render.engine == "CYCLES" and bpy.scene.cycles.device == "GPU"   # re-applied after the factory reset


if __name__ == "__main__":
    test_camera_conversion_looks_at_origin_upright(); test_normalise_and_loop(); print("render_bpy mock tests passed")
