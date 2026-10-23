import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[2] / "scripts/test-tools/Yolo11VietnamPlatePoseTraining/colab_checkpoint_support.py"
spec = importlib.util.spec_from_file_location("colab_checkpoint", TOOL)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = checkpoint.DriveCheckpointStore(self.root / "drive/run", {"dataset": "test", "epochs": 3})
        self.local = self.root / "local"
        self.local.mkdir()
        self.last = self.local / "last.pt"
        self.best = self.local / "best.pt"
        self.last.write_bytes(b"epoch one optimizer and scaler")
        self.best.write_bytes(b"best one")

    def backup(self, epoch=1):
        return self.store.backup(self.last, self.best, epoch, 3)

    def test_rotation_restore_and_completion(self):
        self.backup()
        self.last.write_bytes(b"epoch two optimizer and scaler")
        second = self.backup(2)
        self.assertEqual(second["previous"]["epoch"], 1)
        resumed = checkpoint.DriveCheckpointStore(self.store.root, {"dataset": "test", "epochs": 3}, resume=True)
        path, state = resumed.restore(self.root / "restored")
        self.assertEqual(state["epoch"], 2)
        self.assertEqual(path.read_bytes(), self.last.read_bytes())
        self.assertEqual((path.parent / "weights/best.pt").read_bytes(), self.best.read_bytes())
        self.store.finish()
        with self.assertRaisesRegex(ValueError, "completed"):
            self.store.restore(self.root / "another")
        self.store.restore(self.root / "export-only", allow_completed=True)

    def test_interrupted_copy_does_not_commit_or_destroy_previous(self):
        first = self.backup()
        self.last.write_bytes(b"second")
        original_copy = checkpoint.atomic_copy

        def interrupted(source, destination):
            if Path(source) == self.best:
                raise OSError("synthetic Drive disconnect")
            return original_copy(source, destination)

        with patch.object(checkpoint, "atomic_copy", side_effect=interrupted), self.assertRaises(OSError):
            self.backup(2)
        self.assertEqual(self.store.state(), first)
        _, restored = self.store.restore(self.root / "restored")
        self.assertEqual(restored["epoch"], 1)

    def test_corruption_falls_back_and_repairs_marker_before_next_copy(self):
        self.backup()
        self.last.write_bytes(b"second")
        second = self.backup(2)
        (self.store.root / second["last"]["file"]).write_bytes(b"corrupted")
        _, restored = self.store.restore(self.root / "restored")
        self.assertEqual(restored["epoch"], 1)
        self.assertEqual(self.store.state()["epoch"], 1)
        self.backup(2)
        self.store.checked_path(self.store.state()["previous"]["last"])

    def test_interrupted_marker_commit_keeps_last_committed_checkpoint(self):
        first = self.backup()
        self.last.write_bytes(b"second")
        with patch.object(checkpoint, "atomic_json", side_effect=OSError("synthetic marker disconnect")), self.assertRaises(OSError):
            self.backup(2)
        self.assertEqual(self.store.state(), first)
        _, restored = self.store.restore(self.root / "restored")
        self.assertEqual(restored["epoch"], 1)

    def test_both_generations_corrupted_rejected(self):
        self.backup()
        self.last.write_bytes(b"second")
        state = self.backup(2)
        for entry in (state["last"], state["previous"]["last"]):
            (self.store.root / entry["file"]).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "Both checkpoint"):
            self.store.restore(self.root / "restored")

    def test_run_reuse_and_configuration_mismatch_rejected(self):
        with self.assertRaises(FileExistsError):
            checkpoint.DriveCheckpointStore(self.store.root, {})
        with self.assertRaisesRegex(ValueError, "differs"):
            checkpoint.DriveCheckpointStore(self.store.root, {"dataset": "different"}, resume=True)
        with self.assertRaisesRegex(ValueError, "No committed"):
            self.store.restore(self.root / "restored")

    def test_status_metrics_log_resume_and_heartbeat(self):
        (self.local / "results.csv").write_text("epoch,loss\n1,1.0\n")
        (self.local / "events.out.tfevents.test").write_bytes(b"synthetic event")
        trainer = SimpleNamespace(epoch=0, epochs=3, tloss={"pose_loss": 1.5, "bad": float("nan")},
                                  metrics={"metrics/mAP50-95(P)": .5}, train_loader=[1, 2],
                                  last=self.last, best=self.best, save_dir=self.local)
        dashboard = []
        monitor = checkpoint.TrainingMonitor(self.store, dashboard.append, interval=0)
        monitor.start(trainer)
        monitor.epoch_start(trainer)
        monitor.batch_end(trainer)
        monitor.val_start(None)
        monitor.val_batch_end(None)
        monitor.saved(trainer)
        status = json.loads((self.store.root / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["checkpoint_epoch"], 1)
        self.assertEqual(status["validation_batch"], 1)
        self.assertNotIn("bad", status["losses"])
        self.assertTrue(dashboard)
        self.store.restore(self.root / "restored")
        self.assertTrue((self.root / "restored/results.csv").is_file())
        monitor.finished(trainer)
        self.assertTrue(self.store.state()["completed"])

    def test_support_embedded_in_notebook_matches_source(self):
        notebook = json.loads(TOOL.with_name("Train_YOLO11n_Plate_Pose.ipynb").read_text(encoding="utf-8"))
        source = "".join(next(c for c in notebook["cells"] if c["id"] == "checkpoint-support")["source"])
        self.assertEqual(source, TOOL.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
