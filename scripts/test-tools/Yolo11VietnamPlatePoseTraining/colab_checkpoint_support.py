"""Self-contained code embedded in the Colab notebook; stdlib-only storage core."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    checksum = file_sha256(source)
    if file_sha256(temporary) != checksum:
        raise OSError("Checkpoint copy hash mismatch; previous checkpoint is retained")
    os.replace(temporary, destination)
    return {"file": destination.name, "sha256": checksum, "size": destination.stat().st_size}


class DriveCheckpointStore:
    def __init__(self, run_dir, contract, resume=False):
        self.root = Path(run_dir)
        self.contract = contract
        self.marker = self.root / "checkpoint.json"
        if resume:
            saved = json.loads((self.root / "run-contract.json").read_text(encoding="utf-8"))
            if saved != contract:
                raise ValueError("Run configuration/dataset differs; choose the original configuration or a new RUN_ID")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            atomic_json(self.root / "run-contract.json", contract)

    def state(self):
        return json.loads(self.marker.read_text(encoding="utf-8")) if self.marker.exists() else None

    def checked_path(self, entry):
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Invalid checkpoint path")
        path = self.root / entry["file"]
        if not path.is_file() or path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("Checkpoint is missing or corrupted")
        return path

    def restore(self, local_dir, allow_completed=False):
        state = self.state()
        if not state:
            raise ValueError("No committed checkpoint; epoch 1 has not been backed up yet")
        if state.get("completed") and not allow_completed:
            raise ValueError("Run is already completed; export best.pt or start a new RUN_ID")
        # If the latest slot is damaged, retain the option to resume the previous epoch.
        for snapshot in (state, state.get("previous")):
            if not snapshot:
                continue
            try:
                last = self.checked_path(snapshot["last"])
                best = self.checked_path(snapshot["best"]) if snapshot.get("best") else None
            except (ValueError, OSError):
                continue
            local = Path(local_dir)
            (local / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(last, local / "resume.pt")
            if best:
                shutil.copyfile(best, local / "weights/best.pt")
            if (self.root / "logs").exists():
                shutil.copytree(self.root / "logs", local, dirs_exist_ok=True)
            if snapshot is not state:
                atomic_json(self.marker, snapshot)
            return local / "resume.pt", snapshot
        raise ValueError("Both checkpoint generations failed verification; do not start a fresh run over them")

    def backup(self, last, best, epoch, total_epochs):
        previous = self.state()
        slot = "b" if previous and previous["last"]["file"] == "last-a.pt" else "a"
        last_entry = atomic_copy(last, self.root / f"last-{slot}.pt")
        best_entry = atomic_copy(best, self.root / f"best-{slot}.pt") if Path(best).is_file() else None
        snapshot = dict(epoch=epoch, total_epochs=total_epochs, last=last_entry, best=best_entry,
                        updated_at_utc=datetime.now(timezone.utc).isoformat(), completed=False)
        if previous:
            snapshot["previous"] = {key: value for key, value in previous.items() if key != "previous"}
        # The marker changes only after both complete copies have been verified.
        atomic_json(self.marker, snapshot)
        return snapshot

    def sync_logs(self, local_dir):
        # TensorBoard writes asynchronously. Flush its pinned integration before
        # hashing/copying event files, so a growing file cannot look corrupted.
        integration = sys.modules.get("ultralytics.utils.callbacks.tensorboard")
        writer = getattr(integration, "WRITER", None)
        if writer is not None:
            writer.flush()
        local = Path(local_dir)
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        for file in local.iterdir():
            if file.is_file() and (file.name in {"results.csv", "args.yaml", "environment-freeze.txt", "dataset-manifest.json"}
                                   or file.suffix == ".png" or file.name.startswith("events.out.tfevents")):
                atomic_copy(file, logs / file.name)

    def finish(self):
        snapshot = self.state()
        if not snapshot:
            raise ValueError("No checkpoint committed")
        snapshot["completed"] = True
        atomic_json(self.marker, snapshot)


def finite_metrics(values):
    result = {}
    for key, value in values.items():
        try:
            number = float(value)
            if math.isfinite(number):
                result[str(key)] = round(number, 6)
        except (TypeError, ValueError):
            continue
    return result


class TrainingMonitor:
    def __init__(self, store, display_update=None, interval=20):
        self.store = store
        self.display_update = display_update
        self.interval = interval
        self.last_write = 0
        self.trainer = None
        self.batch = 0
        self.val_batch = 0
        self.epoch_started = time.monotonic()
        self.phase = "initializing"

    def update(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < self.interval:
            return
        trainer = self.trainer
        losses = getattr(trainer, "tloss", {})
        if not hasattr(losses, "items"):
            losses = dict(zip(getattr(trainer, "loss_names", ()), losses if losses is not None else ()))
        checkpoint = self.store.state()
        status = dict(phase=self.phase, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                      epoch=int(getattr(trainer, "epoch", 0)) + 1,
                      total_epochs=int(getattr(trainer, "epochs", 0)),
                      batch=self.batch, train_batches=len(trainer.train_loader) if trainer else 0,
                      validation_batch=self.val_batch, epoch_seconds=round(now - self.epoch_started, 1),
                      checkpoint_epoch=checkpoint["epoch"] if checkpoint else 0,
                      losses=finite_metrics(losses), metrics=finite_metrics(getattr(trainer, "metrics", {})))
        atomic_json(self.store.root / "status.json", status)
        if self.display_update:
            summary = html.escape(json.dumps({**status["losses"], **status["metrics"]}, ensure_ascii=False))
            self.display_update(
                f"<b>{html.escape(self.phase)}</b> | Epoch {status['epoch']}/{status['total_epochs']} | "
                f"Batch {self.batch}/{status['train_batches']} | Validation batch {self.val_batch} | "
                f"{status['epoch_seconds']} s/epoch | Checkpoint Drive: epoch {status['checkpoint_epoch']}<br>"
                f"<progress value='{self.batch}' max='{max(1, status['train_batches'])}'></progress>"
                f"<pre style='white-space:pre-wrap'>{summary}</pre>"
                f"Cập nhật UTC: {status['updated_at_utc']}")
        self.last_write = now

    def start(self, trainer):
        self.trainer = trainer
        self.phase = "training"
        self.update(True)

    def epoch_start(self, trainer):
        self.trainer = trainer
        self.batch = 0
        self.val_batch = 0
        self.phase = "training"
        self.epoch_started = time.monotonic()
        self.update(True)

    def batch_end(self, trainer):
        self.trainer = trainer
        self.batch += 1
        self.update()

    def val_start(self, _validator):
        self.phase = "validating"
        self.val_batch = 0
        self.update(True)

    def val_batch_end(self, _validator):
        self.val_batch += 1
        self.update()

    def saved(self, trainer):
        self.trainer = trainer
        self.phase = "saving_checkpoint"
        self.update(True)
        self.store.backup(trainer.last, trainer.best, int(trainer.epoch) + 1, int(trainer.epochs))
        self.store.sync_logs(trainer.save_dir)
        self.phase = "epoch_saved"
        self.update(True)

    def finished(self, trainer):
        self.trainer = trainer
        self.store.sync_logs(trainer.save_dir)
        self.store.finish()
        self.phase = "completed"
        self.update(True)
