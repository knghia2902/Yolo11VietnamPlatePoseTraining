"""Opt-in CPU integration test; never reads real review images or a product DB."""
import json
import os
from pathlib import Path
import sys
import unittest
import uuid


@unittest.skipUnless(os.environ.get("NATIME_RUN_TRAINING_SMOKE") == "1", "Opt-in pinned Ultralytics CPU smoke")
class TrainingSmoke(unittest.TestCase):
    def test_interrupt_and_resume_optimizer(self):
        import torch
        import ultralytics
        from ultralytics import YOLO, settings
        from PIL import Image, ImageDraw
        tool_dir = Path(__file__).resolve().parents[2] / "scripts/test-tools/Yolo11VietnamPlatePoseTraining"
        sys.path.insert(0, str(tool_dir))
        from colab_checkpoint_support import DriveCheckpointStore, TrainingMonitor
        sys.path.pop(0)
        self.assertEqual(ultralytics.__version__, "8.4.183")
        torch.set_num_threads(1)
        settings.update({"tensorboard": False})
        root = Path("D:/nATime-quarantine/20261022-colab-checkpoint-tests") / ("synthetic-" + uuid.uuid4().hex[:8])
        root.mkdir(parents=True)
        for split, count in (("train", 4), ("val", 2)):
            for index in range(count):
                image = Image.new("RGB", (64, 64), (20 + index * 15, 40, 60))
                ImageDraw.Draw(image).rectangle((12, 20, 52, 44), fill="white")
                images, labels = root / "data/images" / split, root / "data/labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                image.save(images / f"{index}.png")
                (labels / f"{index}.txt").write_text("0 0.5 0.5 0.625 0.375 0.1875 0.3125 2 0.8125 0.3125 2 0.8125 0.6875 2 0.1875 0.6875 2\n")
        data = root / "data.yaml"
        data.write_text(f"path: {(root/'data').as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [4, 3]\nflip_idx: []\nnames:\n  0: plate\n")
        contract = {"dataset": "synthetic-only", "epochs": 2}
        store = DriveCheckpointStore(root / "drive/run", contract)
        monitor = TrainingMonitor(store)
        model = YOLO("yolo11n-pose.yaml")
        model.add_callback("on_train_start", monitor.start)
        model.add_callback("on_train_epoch_start", monitor.epoch_start)

        def interrupt(trainer):
            monitor.saved(trainer)
            raise RuntimeError("synthetic-disconnect-after-checkpoint")

        model.add_callback("on_model_save", interrupt)
        with self.assertRaisesRegex(RuntimeError, "^synthetic-disconnect-after-checkpoint$"):
            model.train(data=str(data), epochs=2, imgsz=64, batch=2, device="cpu", workers=0,
                        amp=False, plots=False, pretrained=False, optimizer="SGD", mosaic=0.,
                        fliplr=0., flipud=0., project=str(root / "local"), name="run", exist_ok=True, save=True)
        self.assertEqual(store.state()["epoch"], 1)
        restored = DriveCheckpointStore(root / "drive/run", contract, resume=True)
        checkpoint, _ = restored.restore(root / "restored/run")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertIsNotNone(payload["optimizer"])
        self.assertEqual(payload["epoch"], 0)
        new_data = root / "reconnected-data.yaml"
        new_data.write_text(data.read_text())
        model = YOLO(str(checkpoint))
        monitor = TrainingMonitor(restored)
        for event, callback in (("on_train_start", monitor.start), ("on_train_epoch_start", monitor.epoch_start),
                                ("on_model_save", monitor.saved), ("on_train_end", monitor.finished)):
            model.add_callback(event, callback)
        model.train(resume=True, data=str(new_data), device="cpu", workers=0, batch=2, save_dir=str(root / "restored/run"))
        self.assertEqual(model.trainer.start_epoch, 1)
        self.assertEqual(Path(model.trainer.args.data).resolve(), new_data.resolve())
        self.assertEqual(Path(model.trainer.save_dir).resolve(), (root / "restored/run").resolve())
        state = restored.state()
        self.assertEqual(state["epoch"], 2)
        self.assertTrue(state["completed"])
        payload = torch.load(restored.checked_path(state["last"]), map_location="cpu", weights_only=False)
        self.assertIsNotNone(payload["optimizer"], "Drive checkpoint must remain resumable after final local weights are stripped")
        report = dict(ultralytics=ultralytics.__version__, synthetic_only=True, interrupted_epoch=1,
                      resumed_epoch=2, optimizer_preserved=True, new_dataset_and_save_paths_applied=True, completed=True)
        (root / "smoke-report.json").write_text(json.dumps(report, indent=2))
        print("CHECKPOINT_RESUME_SMOKE", root / "smoke-report.json")


if __name__ == "__main__":
    unittest.main()
