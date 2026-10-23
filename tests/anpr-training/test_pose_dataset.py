"""Synthetic-only tests: no customer images or network required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

TOOL = Path(__file__).resolve().parents[2] / "scripts/test-tools/Yolo11VietnamPlatePoseTraining/prepare_pose_dataset.py"
spec = importlib.util.spec_from_file_location("pose_prepare", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
POINTS = [[.2, .2], [.8, .25], [.75, .8], [.25, .75]]


def image_bytes(reverse=False):
    image = Image.new("RGB", (80, 60))
    for y in range(60):
        for x in range(80):
            image.putpixel((x, y), ((255 - x * 3) if reverse else x * 3, y * 4, 100))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def scan(self, duplicate=False, invalid=False):
        source = self.root / "raw.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for name, data in [("a_1", image_bytes()), ("b_1", image_bytes(True))] + ([("c_1", image_bytes())] if duplicate else []):
                archive.writestr("images/" + name + ".png", data)
                archive.writestr("labels/" + name + ".txt", "0 1 2\n" if invalid else tool.label_text(POINTS))
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = tool.scan(source, self.root / "workspace")
        return manifest

    def reviews(self, manifest):
        for record in manifest["records"]:
            record.update(reviewed=True, group_reviewed=True, reviewer="synthetic tester")
        path = self.root / "reviews.json"
        tool.write_json(path, manifest)
        return path

    def finalize(self, manifest):
        with contextlib.redirect_stdout(io.StringIO()):
            return tool.finalize(self.root / "workspace", self.reviews(manifest), self.root / "package")

    def test_pose_roundtrip(self):
        box, points = tool.parse_label(tool.label_text(POINTS))
        self.assertEqual(points, POINTS)
        self.assertAlmostEqual(box[2], .6)

    def test_invalid_geometry_and_numbers_rejected(self):
        for points in [POINTS[::-1], [POINTS[i] for i in (0, 2, 1, 3)], [[0, 0]] * 4,
                       [[float("nan"), .2]] + POINTS[1:], [[-1, .2]] + POINTS[1:], POINTS[1:] + POINTS[:1]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                tool.valid_points(points)

    def test_missing_and_multiobject_labels_rejected(self):
        for text in ["", tool.label_text(POINTS) * 2, "0 1 2"]:
            with self.assertRaises(ValueError):
                tool.parse_label(text)

    def test_scan_deduplicates_and_retains_provenance(self):
        manifest = self.scan(duplicate=True)
        self.assertEqual(manifest["unique_images"], 2)
        self.assertEqual(manifest["removed_duplicates"], 1)
        self.assertEqual(manifest["records"][0]["source_groups"], ["a", "c"])
        self.assertFalse(manifest["records"][0]["reviewed"])

    def test_scan_errors_block_export(self):
        manifest = self.scan(invalid=True)
        self.assertEqual(len(manifest["errors"]), 2)
        with self.assertRaisesRegex(ValueError, "Scan errors"):
            self.finalize(manifest)

    def test_unreviewed_corners_block_export(self):
        manifest = self.scan()
        path = self.root / "unreviewed.json"
        tool.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            tool.finalize(self.root / "workspace", path, self.root / "package")
        self.assertFalse((self.root / "package").exists())

    def test_wrong_source_and_duplicate_review_ids_rejected(self):
        manifest = self.scan()
        path = self.reviews(manifest)
        manifest["source_sha256"] = "bad"
        tool.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "different source"):
            tool.finalize(self.root / "workspace", path, self.root / "package")

    def test_edited_image_after_review_rejected(self):
        manifest = self.scan()
        (self.root / "workspace" / manifest["records"][0]["image"]).write_bytes(image_bytes(True))
        with self.assertRaisesRegex(ValueError, "Image changed"):
            self.finalize(manifest)

    def test_export_verify_and_tamper_detection(self):
        manifest = self.scan()
        package = self.finalize(manifest)
        self.assertEqual({r["split"] for r in package["records"]}, {"train", "val"})
        self.assertTrue((self.root / "package-training.zip").exists())
        label = self.root / "package" / package["records"][0]["label"]
        label.write_text(tool.label_text([[.1, .1], [.9, .1], [.9, .9], [.1, .9]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            tool.verify(self.root / "package")

    def test_unexpected_files_rejected(self):
        self.finalize(self.scan())
        (self.root / "package/images/train/extra.png").write_bytes(image_bytes())
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            tool.verify(self.root / "package")

    def test_notebook_preflight_accepts_real_exported_archive(self):
        self.finalize(self.scan())
        notebook = json.loads(TOOL.with_name("Train_YOLO11n_Plate_Pose.ipynb").read_text(encoding="utf-8"))
        code = "".join(next(cell for cell in notebook["cells"] if cell["id"] == "pose-03")["source"])
        code = code.replace("from google.colab import files", "")
        code = code.replace("dir='/content'", "dir=TEST_ROOT")
        archive_path = str(self.root / "package-training.zip")
        files = type("Files", (), {"upload": staticmethod(lambda: {archive_path: b""})})
        cache = self.root / "cached-dataset-training.zip"
        scope = {"files": files, "Path": Path, "TEST_ROOT": str(self.root), "DRIVE_DATASET": cache}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "notebook-preflight", "exec"), scope)
        self.assertTrue(scope["DATA"].is_file())
        self.assertEqual(len(scope["manifest"]["records"]), 2)
        self.assertTrue(cache.is_file())
        files.upload = staticmethod(lambda: self.fail("Cached dataset must not ask for reupload"))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "notebook-cached-preflight", "exec"), scope)

    def test_group_and_near_duplicate_transitive_components(self):
        rows = [dict(id=str(i), group=g, source_groups=[g], dhash=h)
                for i, (g, h) in enumerate([("a", 0), ("b", 1), ("b", 255), ("c", (1 << 64) - 1)])]
        split = tool.split_groups(rows, distance=1)
        self.assertEqual(split["0"], split["1"])
        self.assertEqual(split["1"], split["2"])
        self.assertNotEqual(split["0"][0], split["3"][0])
        self.assertEqual(split, tool.split_groups(list(reversed(rows)), distance=1))

    def test_single_component_fails_closed(self):
        rows = [dict(id=str(i), group="same", source_groups=["same"], dhash=i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "independent"):
            tool.split_groups(rows)

    def test_output_cannot_overwrite_or_enter_repo(self):
        with self.assertRaises(ValueError):
            tool.fresh_output(TOOL.parent / "dataset")
        with self.assertRaises(FileExistsError):
            tool.fresh_output(self.root)


if __name__ == "__main__":
    unittest.main()
