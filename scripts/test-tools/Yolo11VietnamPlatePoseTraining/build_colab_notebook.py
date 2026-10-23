"""Rebuild the self-contained Colab notebook without changing the dataset ZIP."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "Train_YOLO11n_Plate_Pose.ipynb"


def main():
    notebook = json.loads(TARGET.read_text(encoding="utf-8"))
    original = {cell["id"]: cell for cell in notebook["cells"]}
    cells = []

    def add(identifier, kind, source):
        cell = dict(id=identifier, cell_type=kind, metadata={}, source=source.splitlines(True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def keep(identifier):
        cells.append(original[identifier])

    keep("pose-00")
    add("pose-01", "code", """%pip install ultralytics==8.4.183 Pillow==11.3.0 onnx onnxruntime onnxslim tensorboard
import sys, subprocess
from pathlib import Path
subprocess.run(['nvidia-smi'], check=True)
with open('/content/environment-freeze.txt', 'w') as environment:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, stdout=environment)
""")
    add("drive-help", "markdown", """## Google Drive và cấu hình chạy
Cho phép Colab kết nối Drive của bạn. Dataset được cache trên Drive; ảnh được giải nén ở ổ local Colab để train nhanh.

- Lần đầu: `RESUME=False`, `RUN_ID=''`. Chạy smoke trước với `FULL_TRAIN=False`.
- Train đầy đủ: sửa `FULL_TRAIN=True`, giữ `RESUME=False`, `RUN_ID=''`, chạy lại từ ô này.
- Sau khi ngắt: chạy lại notebook, đặt đúng `FULL_TRAIN` như lần trước, `RESUME=True`, dán **RUN_ID cũ**. Dataset cache được dùng lại; tiếp tục từ epoch đã backup.
- Không chạy hai phiên Colab đồng thời vào cùng RUN_ID. Khi tạo run mới, thư mục Drive đã tồn tại sẽ bị từ chối để tránh ghi đè.

Mỗi epoch hoàn tất được backup sang hai slot luân phiên, kiểm tra SHA-256 rồi mới cập nhật checkpoint.json. Slot trước vẫn còn nếu quá trình copy bị ngắt. Batch/epoch đang chạy dở có thể phải chạy lại; trước checkpoint đầu tiên chưa có tiến độ có thể phục hồi.
""")
    add("drive-setup", "code", """from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import uuid, re
drive.mount('/content/drive')

FULL_TRAIN = False
RESUME = False
RUN_ID = ''  # Khi resume: dán chính xác RUN_ID được in trong lần chạy trước.
IMGSZ = 960
BATCH = 4
EPOCHS = 100 if FULL_TRAIN else 2
DRIVE_ROOT = Path('/content/drive/MyDrive/nATime-YOLO11-Pose')
DRIVE_DATASET = DRIVE_ROOT / 'datasets/dataset-training.zip'
DRIVE_DATASET.parent.mkdir(parents=True, exist_ok=True)
if RESUME:
    assert RUN_ID, 'Cần RUN_ID cũ để tiếp tục'
elif not RUN_ID:
    RUN_ID = ('full-' if FULL_TRAIN else 'smoke-') + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
assert re.fullmatch(r'[A-Za-z0-9_-]{1,100}', RUN_ID), 'RUN_ID chỉ dùng chữ, số, gạch ngang/gạch dưới'
DRIVE_RUN = DRIVE_ROOT / 'runs' / RUN_ID
LOCAL_RUN = Path('/content/natime-pose-runs') / RUN_ID
LOCAL_RUN.mkdir(parents=True, exist_ok=True)
print('RUN_ID:', RUN_ID)
print('Drive:', DRIVE_RUN)
print('Mode:', 'FULL' if FULL_TRAIN else 'SMOKE', '| Resume:', RESUME)
""")
    keep("pose-02")
    source = "".join(original["pose-03"]["source"])
    # Preserve the verifier pin and safe ZIP checks from the prepared package.
    start = source.index("ROOT = Path(tempfile.mkdtemp")
    source = source.split("\nif not DRIVE_DATASET.exists():", 1)[0]
    source = """from google.colab import files
import zipfile, hashlib, importlib.util, json, tempfile, shutil, os
if DRIVE_DATASET.is_file():
    local_archive = Path(tempfile.mkdtemp(prefix='pose-zip-', dir='/content')) / 'dataset-training.zip'
    shutil.copyfile(DRIVE_DATASET, local_archive)
    archives = [str(local_archive)]
    print('Dùng dataset đã cache trên Drive')
else:
    uploaded = files.upload()
    archives = [name for name in uploaded if name.endswith('-training.zip')]
    assert len(archives) == 1, 'Chọn đúng một ZIP đã duyệt *-training.zip'
""" + source[start:]
    source += """
if not DRIVE_DATASET.exists():
    temporary = DRIVE_DATASET.with_suffix('.zip.tmp')
    shutil.copyfile(archives[0], temporary)
    with open(archives[0], 'rb') as stream:
        original_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with temporary.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == original_hash
    os.replace(temporary, DRIVE_DATASET)
    print('Dataset đã được lưu vào Drive')
"""
    add("pose-03", "code", source)
    keep("pose-04")
    keep("pose-05")
    add("support-help", "markdown", "## Checkpoint và bảng hoạt động\nChạy ô mã hỗ trợ bên dưới trước khi train. Đây là mã đi kèm notebook, không cần tải thêm file Python.\n")
    add("checkpoint-support", "code", (ROOT / "colab_checkpoint_support.py").read_text(encoding="utf-8"))
    add("tensorboard-help", "markdown", """## Theo dõi trực tiếp
Chạy ô TensorBoard **trước** ô train, rồi cuộn lại bảng này trong khi training. Xem Scalars: train/pose_loss, val/pose_loss và metrics/mAP50-95(P) (tên có thể khác theo phiên bản). Log Ultralytics trong ô train vẫn hiển thị epoch, batch, GPU memory và thời gian.

Bảng trạng thái ở ô train cập nhật khoảng 20 giây/lần và tại các mốc epoch/validation/checkpoint. `checkpoint_epoch` là epoch đã lưu lên Drive, khác với epoch đang tính. `status.json` chỉ là dấu hoạt động gần nhất; file cũ không chứng minh runtime vẫn đang chạy.
""")
    add("tensorboard", "code", """from ultralytics import settings
settings.update({'tensorboard': True})
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir /content/natime-pose-runs')
""")
    keep("pose-06")
    add("pose-07", "code", """from ultralytics import YOLO, __version__
from IPython.display import display, HTML
import torch, shutil, json, hashlib
assert torch.cuda.is_available(), 'Cần GPU Colab'
assert __version__ == '8.4.183', 'Dùng đúng phiên bản đã pin để resume nhất quán'
prepare.verify(ROOT)
contract = dict(manifest_sha256=file_sha256(ROOT/'manifest.json'),
                full_train=FULL_TRAIN, epochs=EPOCHS, imgsz=IMGSZ,
                kpt_shape=[4,3], ultralytics=__version__)
store = DriveCheckpointStore(DRIVE_RUN, contract, resume=RESUME)
already_finished = False
if RESUME:
    checkpoint, saved = store.restore(LOCAL_RUN, allow_completed=True)
    already_finished = saved.get('completed', False) or saved['epoch'] >= EPOCHS
    model = YOLO(str(checkpoint))
    print('Checkpoint Drive đã xác minh:', saved['epoch'], '/', saved['total_epochs'])
else:
    model = YOLO('yolo11n-pose.pt')

shutil.copyfile('/content/environment-freeze.txt', LOCAL_RUN/'environment-freeze.txt')
shutil.copyfile(ROOT/'manifest.json', LOCAL_RUN/'dataset-manifest.json')
atomic_copy(LOCAL_RUN/'environment-freeze.txt', DRIVE_RUN/'environment-freeze.txt')
atomic_copy(LOCAL_RUN/'dataset-manifest.json', DRIVE_RUN/'dataset-manifest.json')
panel = display(HTML('Đang chuẩn bị training...'), display_id=True)
monitor = TrainingMonitor(store, lambda markup: panel.update(HTML(markup)))
for event, callback in {
    'on_train_start': monitor.start,
    'on_train_epoch_start': monitor.epoch_start,
    'on_train_batch_end': monitor.batch_end,
    'on_val_start': monitor.val_start,
    'on_val_batch_end': monitor.val_batch_end,
    'on_model_save': monitor.saved,
    'on_train_end': monitor.finished,
}.items():
    model.add_callback(event, callback)

if already_finished:
    panel.update(HTML('Lượt train đã hoàn tất. Có thể chạy ô export.'))
    print('Run đã hoàn tất. Có thể chạy ô export; không train lại.')
else:
    try:
        if RESUME:
            # Pinned trainer restores optimizer/scaler/EMA and updates the new dataset YAML path.
            results = model.train(resume=True, data=str(DATA), device=0,
                                  batch=BATCH, workers=2, save_dir=str(LOCAL_RUN))
        else:
            results = model.train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                                  device=0, workers=2, seed=42, deterministic=True,
                                  fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
                                  degrees=0.0, perspective=0.0, patience=20,
                                  project=str(LOCAL_RUN.parent), name=RUN_ID,
                                  exist_ok=True, save=True, save_period=-1, plots=True)
    except BaseException:
        monitor.phase = 'interrupted'
        try:
            monitor.update(True)
        except Exception:
            print('Không ghi được status lên Drive. Kiểm tra checkpoint.json trước khi resume.')
        raise
RUN = LOCAL_RUN
print('RUN_ID:', RUN_ID, '| Drive checkpoints:', DRIVE_RUN)
print('Đã lưu:', store.state())
""")
    add("status-help", "markdown", """## Xem trạng thái đã lưu / sau khi reconnect
Sau khi mount Drive và chọn đúng RUN_ID, có thể chạy ô này để đọc lần hoạt động gần nhất. Khi Colab bị ngắt, chạy lại các ô thiết lập và đặt RESUME=True; TensorBoard sẽ đọc cả log cũ được phục hồi. Nếu run đã hoàn tất thì notebook bỏ qua train và cho export.
""")
    add("status-view", "code", """import json
from datetime import datetime, timezone
status_path = DRIVE_RUN/'status.json'
if status_path.exists():
    status = json.loads(status_path.read_text(encoding='utf-8'))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(status['updated_at_utc'])).total_seconds()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f'Lần cập nhật gần nhất cách đây {age:.0f} giây; đây không phải kiểm tra runtime còn sống.')
else:
    print('Chưa có trạng thái hoạt động cho RUN_ID này.')
marker = DRIVE_RUN/'checkpoint.json'
if marker.exists():
    print('Checkpoint đã commit:', json.loads(marker.read_text(encoding='utf-8')))
""")
    keep("pose-08")
    export = "".join(original["pose-09"]["source"]).split("\n# Keep exports on Drive too;", 1)[0]
    export += """
# Keep exports on Drive too; resumable training checkpoints stay unchanged.
exports = DRIVE_RUN/'exports'
exports.mkdir(exist_ok=True)
for artifact in (target, RUN/'model-contract.json', RUN/'environment-freeze.txt', best):
    atomic_copy(artifact, exports/artifact.name)
print('Đã lưu model và báo cáo vào:', exports)
"""
    add("pose-09", "code", export)
    notebook["cells"] = cells
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
