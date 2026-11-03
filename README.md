# YOLO11 Vietnam Plate Pose Training

Repository độc lập để **chuẩn bị dữ liệu, duyệt bốn góc biển số Việt Nam, train YOLO11 Point và test model trên web**. OCR biển số là luồng riêng, không phải mục tiêu của lượt train góc.

Model dùng **YOLO11n-Pose pretrained**, fine-tune thành đầu ra `kpt_shape: [4, 3]`. Thứ tự bốn điểm:

**1. Trên trái (TL) → 2. Trên phải (TR) → 3. Dưới phải (BR) → 4. Dưới trái (BL).**

## Kết quả lượt huấn luyện ngày 02–03/11/2026

- Dữ liệu gồm **10.000 ảnh huấn luyện** và **1.500 ảnh validation** riêng. Ảnh được kiểm tra, gán nhãn vùng biển số và bốn góc; cần tiếp tục xác minh quyền sử dụng theo từng nguồn.
- Model nền: **YOLO11n-Pose pretrained**, fine-tune thành một lớp biển số và bốn điểm góc; sử dụng Ultralytics 8.4.183, kích thước ảnh 960 và GPU Tesla T4.
- Cấu hình tối đa 100 epoch; quá trình dừng sớm sau **81 epoch**, khoảng **12,65 giờ**, do không cải thiện trong 20 epoch. Checkpoint tốt nhất tại **epoch 61**.
- Trên tập validation 1.500 ảnh, kết quả của checkpoint tốt nhất: **box mAP@50: 0,995; box mAP@50–95: 0,904; pose mAP@50: 0,995; pose mAP@50–95: 0,995**. Đây là chỉ số đánh giá mô hình, không phải tỷ lệ OCR đọc đúng hay hiệu quả thực tế tại cổng.
- Model dự đoán bốn góc để căn chỉnh phối cảnh trước OCR. Cần tiếp tục đánh giá trên dữ liệu thực tế độc lập, gồm biển nghiêng, hai dòng, mờ và thiếu sáng, cũng như tốc độ và tỷ lệ đọc đúng toàn bộ chuỗi OCR.
- Model ONNX epoch 61 dùng cho Gate: [models/yolo11-point-epoch61.onnx](models/yolo11-point-epoch61.onnx); SHA-256 `f678438d03dd08b2c68c477d9bdaead25b2852f643731dec5ee7c3594ae49fa7`. Checkpoint và dữ liệu ảnh không lưu trong Git.
- Metadata model ghi AGPL-3.0. Model sử dụng thành phần Ultralytics; cần xem xét giấy phép hoặc thỏa thuận phù hợp trước khi phân phối ứng dụng đóng mã nguồn. Quyền sử dụng dữ liệu là vấn đề riêng.

## Mục lục

- [1. Quy trình tổng thể](#1-quy-trình-tổng-thể)
- [2. Source web và train nằm ở đâu?](#2-source-web-và-train-nằm-ở-đâu)
- [3. Cài môi trường Windows](#3-cài-môi-trường-windows)
- [4. Tổ chức dữ liệu và workspace](#4-tổ-chức-dữ-liệu-và-workspace)
- [5. Chạy web](#5-chạy-web)
- [6. Duyệt góc và chia validation](#6-duyệt-góc-và-chia-validation)
- [7. Chuẩn bị gói train](#7-chuẩn-bị-gói-train)
- [8. Train trên Colab](#8-train-trên-colab)
- [9. Đọc chỉ số và resume](#9-đọc-chỉ-số-và-resume)
- [10. Đăng ký và test model mới](#10-đăng-ký-và-test-model-mới)
- [11. Xử lý lỗi](#11-xử-lý-lỗi)
- [12. Tests và tài liệu](#12-tests-và-tài-liệu)

## 1. Quy trình tổng thể

| Bước | Thao tác | Kết quả |
|---|---|---|
| 1 | Thu thập ảnh, kiểm tra nguồn, loại ảnh không phù hợp và trùng | Tập ảnh gốc |
| 2 | Chỉnh, kiểm tra và duyệt bốn góc trên web | Ảnh và nhãn YOLO Pose đúng góc thật |
| 3 | Tách validation theo nhóm xe/ảnh trùng | Hai tập độc lập |
| 4 | Chốt snapshot, kiểm tra hash/nhãn, đóng gói ZIP | Gói khớp notebook |
| 5 | Chạy thử rồi train đầy đủ trên Colab | Checkpoint và best model trên Drive |
| 6 | Tải `best.pt`, đăng ký version vào web | Model mới trong danh sách |
| 7 | Test ảnh mới và kiểm tra bằng mắt | Đánh giá lỗi bỏ sót, nhận nhầm, lệch góc |

**Duyệt góc đạt nghĩa là nhãn góc đủ yêu cầu train YOLO Point; không bắt buộc đã định danh biển số bằng OCR.** Trước khi đưa vào snapshot vẫn phải kiểm tra trùng và phân chia validation.

Git lưu source, test, notebook sạch, hướng dẫn và model ONNX đã chọn. **Ảnh thực, ZIP dataset, SQLite, manifest dữ liệu thực, checkpoint, credentials và log giữ riêng ngoài Git.** Clone repository chưa đủ để tái chạy một snapshot nếu chưa có dữ liệu riêng tương ứng.

## 2. Source web và train nằm ở đâu?

| Thành phần | File |
|---|---|
| Khởi chạy web từ root | [Start-Web.ps1](Start-Web.ps1) |
| Dependencies web | [requirements-web.txt](requirements-web.txt) |
| Dependencies inference | [requirements-inference.txt](requirements-inference.txt) |
| Backend web duyệt và Test model | [pose_model_test_server.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_server.py) |
| Giao diện Train YOLO point / OCR | [review_expansion.html](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_expansion.html) |
| Giao diện Test model | [review_model_test.html](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_model_test.html) |
| Worker dự đoán và nắn biển | [pose_model_test_worker.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_worker.py) |
| Đăng ký version model | [register_test_model.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/register_test_model.py) |
| Quản lý dữ liệu mở rộng | [curate_acs_training_pool.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/curate_acs_training_pool.py) |
| Đồng bộ nhãn và folder train | [review_train_storage.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_train_storage.py) |
| Chuyển nhóm ô tô/xe máy | [review_vehicle_storage.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_vehicle_storage.py) |
| Lưu/khôi phục checkpoint Drive | [colab_checkpoint_support.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/colab_checkpoint_support.py) |
| Notebook 10.000 train / 1.500 validation | [Train_YOLO11_Point_10000.ipynb](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11_Point_10000.ipynb) |
| Xác minh snapshot 10.000/1.500 | [verify_dataset.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/verify_dataset.py) |
| Automated tests | [tests/anpr-training](tests/anpr-training) |

Web có backend Python và dữ liệu riêng; **không chạy bằng GitHub Pages**. GitHub lưu source; chạy server trên máy chứa workspace và model.

## 3. Cài môi trường Windows

Các lệnh dùng **PowerShell**, chạy từ root repository. Ví dụ dùng Python 3.13 cho môi trường inference CPU đã kiểm tra. OCR cũ có môi trường Python 3.12 riêng; không cần cài OCR khi chỉ test góc.

### 3.1. Clone source

```powershell
git clone https://github.com/knghia2902/Yolo11VietnamPlatePoseTraining.git D:/Yolo11VietnamPlatePoseTraining
Set-Location D:/Yolo11VietnamPlatePoseTraining
```

Nếu đã clone, dùng `git pull --ff-only` khi không có thay đổi local cần giữ.

### 3.2. Môi trường web

```powershell
python -m venv D:/ANPR-Runtime/web-venv
& D:/ANPR-Runtime/web-venv/Scripts/python.exe -m pip install -r requirements-web.txt
```

### 3.3. Môi trường inference CPU

```powershell
python -m venv D:/ANPR-Runtime/inference-venv
& D:/ANPR-Runtime/inference-venv/Scripts/python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
& D:/ANPR-Runtime/inference-venv/Scripts/python.exe -m pip install -r requirements-inference.txt
& D:/ANPR-Runtime/inference-venv/Scripts/python.exe -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__)"
```

Tách inference khỏi web để thay thư viện không ảnh hưởng luồng duyệt. Không trộn torch CPU với torchvision của bản CUDA khác. Worker web hiện dùng CPU, `imgsz=960`, `conf=0.25`; cài CUDA không tự chuyển worker sang GPU. Notebook cài dependencies train riêng trên Colab.

## 4. Tổ chức dữ liệu và workspace

Ví dụ bố trí ngoài repository:

```text
D:/ANPR-Training-Data/
  anh-goc/
    ACS/o-to/
    ACS/xe-may/
    Roboflow/o-to/
    Roboflow/xe-may/
  du-lieu-yolo-point/
    train/
      images/o-to/
      images/xe-may/
      labels/o-to/
      labels/xe-may/
    validation/
      images/o-to/
      images/xe-may/
      labels/o-to/
      labels/xe-may/
  model-yolo-point/
    run-20261103/
      best.pt
      model.json
D:/ANPR-Workspace/expansion/
  curation.sqlite
  previews/
  crops/
D:/ANPR-Secrets/
  review.json
```

- **Ảnh gốc:** ảnh đang chờ xử lý/duyệt.
- **Train:** ảnh và nhãn đã duyệt, có thể đưa vào snapshot sau kiểm tra trùng.
- **Validation:** tập đánh giá riêng, không trùng train.
- **Workspace:** database, đường dẫn ảnh, lịch sử duyệt và preview. Đây không chỉ là folder ảnh.
- **Registry model:** mỗi lượt train một version; giữ model cũ để so sánh.

`images/xe-may/abc.jpg` phải có nhãn tương ứng `labels/xe-may/abc.txt` trong cùng split. Nên move/đổi nhóm trên web để đồng bộ ảnh, nhãn và đường dẫn database. Move/xóa bằng Explorer có thể làm dữ liệu web không còn khớp.

Luồng duyệt hiện tại move ảnh đạt vào vùng train/validation theo cấu hình storage; không tạo thêm bản sao ảnh để duyệt. ZIP snapshot là bản xuất riêng cho train.

**Máy mới chưa có workspace:** xem [chuẩn bị dữ liệu](docs/anpr-pose-training-preparation.md) và [luồng mở rộng](docs/anpr-vietnam-expansion.md). Lệnh `init` của tool mở rộng yêu cầu bộ nền đúng schema; không trỏ vào folder ảnh trống hoặc mặc định coi ZIP 10.000/1.500 cùng schema v3. Giữ workspace đang dùng hoặc chuẩn bị đúng schema trước khi chạy web.

## 5. Chạy web

### 5.1. Credentials

Tạo JSON ngoài checkout, ví dụ `D:/ANPR-Secrets/review.json`:

```json
{
  "username": "reviewer",
  "password": "THAY_BANG_MAT_KHAU_RIENG_CUA_BAN"
}
```

Phải thay password mẫu. Không commit credentials, đặt trong URL hoặc chia sẻ qua ảnh chụp. Server Test model đọc tệp đã tồn tại, không tự sinh credentials.

### 5.2. Chạy trên máy chủ

```powershell
.\Start-Web.ps1 `
  -Python D:/ANPR-Runtime/web-venv/Scripts/python.exe `
  -Workspace D:/ANPR-Workspace/expansion `
  -Credentials D:/ANPR-Secrets/review.json `
  -ModelRoot D:/ANPR-Training-Data/model-yolo-point `
  -InferencePython D:/ANPR-Runtime/inference-venv/Scripts/python.exe
```

Mở `http://127.0.0.1:8767/`, đăng nhập bằng credentials đã cấu hình. Trang test trực tiếp là `/model-test`. Registry chưa có model thì đăng ký theo mục 10.

### 5.3. Cho máy khác trong LAN truy cập

```powershell
.\Start-Web.ps1 `
  -Python D:/ANPR-Runtime/web-venv/Scripts/python.exe `
  -Workspace D:/ANPR-Workspace/expansion `
  -Credentials D:/ANPR-Secrets/review.json `
  -ModelRoot D:/ANPR-Training-Data/model-yolo-point `
  -InferencePython D:/ANPR-Runtime/inference-venv/Scripts/python.exe `
  -Bind 0.0.0.0 -Port 8767 -AllowNetwork 10.32.0.0/16
```

Máy khác mở `http://<IP-may-chu>:8767/`; ví dụ `http://10.32.0.180:8767/` nếu đó là IP thật. Không truy cập bằng `0.0.0.0`. Thay dải mạng cho đúng LAN và kiểm tra firewall. Basic Auth qua HTTP chỉ nên dùng trong mạng tin cậy; cần lớp HTTPS khi triển khai vượt phạm vi này.

Giữ terminal mở; `Ctrl+C` dừng server. Chỉ chạy một server trên một cổng. Nếu khởi động nền bằng `Start-Process`, dùng `-WindowStyle Hidden` và lưu log ngoài checkout. Entry point `curate_acs_training_pool.py serve` cũ không có trang Test model; dùng launcher trên.

## 6. Duyệt góc và chia validation

### Tab Train YOLO point

1. Nhập người duyệt, chọn nguồn và bộ lọc.
2. Kiểm tra bốn góc của **mép biển thật**, không lấy bốn góc khung bao chữ nhật thay cho biển chụp chéo.
3. Kéo điểm TL → TR → BR → BL. Xem crop nắn để kiểm tra độ méo.
4. Chọn đích **Train** hoặc **Validation**, xác nhận một biển Việt Nam và thấy đủ bốn góc, rồi duyệt.
5. Ảnh chưa chắc chắn đưa về **Góc cần bạn kiểm tra**, không duyệt để chạy đủ số lượng.

| Phím/nút | Tác dụng |
|---|---|
| `1` | Duyệt góc đạt vào đích đang chọn rồi chuyển ảnh |
| `2` | Đưa về Góc cần bạn kiểm tra rồi chuyển ảnh |
| `←` / `→` | Chuyển ảnh trước/sau |
| `Enter` | Duyệt tương ứng tab và trạng thái hiện tại |
| Hoàn tác duyệt góc | Hoàn tác lượt duyệt từ lịch sử trình duyệt |

Phím tắt không chạy khi đang nhập text/chọn control; ảnh khóa có giới hạn riêng. Hoàn tác cần phiên bản database khớp lượt đã lưu. Vị trí duyệt được nhớ theo tab/bộ lọc trong trình duyệt; xóa localStorage hoặc đổi trình duyệt có thể mất vị trí nhớ.

### Tab OCR biển số

Kiểm tra nội dung biển và đối chiếu định danh/trùng xe. Không bắt buộc hoàn tất OCR để duyệt nhãn góc. Không coi OCR đề xuất là nhãn thật khi chưa kiểm tra.

### Validation

- Tách cả ảnh và nhãn khỏi train; cùng ảnh không được ở hai phía.
- Kiểm tra hash file, pixel, ảnh gần trùng và nhóm cùng xe/lượt chụp. Đổi tên file không loại được trùng.
- Giữ validation cố định để so các model. Không liên tục chọn lại validation theo kết quả model.
- Nếu model từng học ảnh được chuyển sang validation, đánh giá model đó không còn độc lập.

Snapshot gần đây có **10.000 train + 1.500 validation**. Đây không phải bộ đếm dữ liệu live hoặc yêu cầu cố định cho mọi bộ ảnh.

## 7. Chuẩn bị gói train

| Notebook | Phạm vi |
|---|---|
| [10.000/1.500 trên Colab](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11_Point_10000.ipynb) | Snapshot 10.000 train / 1.500 validation đã chốt |
| [Mixed v3](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb) | Snapshot lịch sử có validation ô tô + xe máy |
| [Mixed v2](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb) | Snapshot lịch sử v2 |
| [Notebook ban đầu](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose.ipynb) | Bộ source-heldout ban đầu |

Notebook gắn với schema, manifest và checksum của gói tương ứng; không hoán đổi ZIP giữa các phiên bản.

Gói riêng 10.000/1.500 có tên `ANPR_YOLO11_Point_10000_train_1500_val.zip`. Bên trong cần ảnh/nhãn, `manifest.json`, `data.yaml`, `verify_dataset.py`, `colab_checkpoint_support.py` và weights khởi tạo đúng checksum.

Kiểm tra snapshot đã giải nén và checksum ZIP:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/verify_dataset.py D:/private/snapshot-10000-1500
Get-FileHash D:/private/ANPR_YOLO11_Point_10000_train_1500_val.zip -Algorithm SHA256
```

Notebook kiểm tra SHA-256 và manifest trước train. Gói dữ liệu thực không đi kèm Git. **Không bỏ assert checksum khi báo sai gói.**

Khi tạo dataset mới, cần chốt split, nhãn, manifest, weights khởi tạo và checksum rồi tạo notebook khớp gói. Packager v3 phục vụ schema v3 lịch sử; hiện chưa có một lệnh tổng quát tự xuất mọi workspace sang đúng schema 10.000/1.500. Không chỉ sửa con số/checksum rồi coi bộ mới đã được kiểm tra.

## 8. Train trên Colab

1. Mở đúng notebook và lưu một bản vào Drive nếu cần giữ cấu hình/kết quả.
2. Chọn **Runtime → Change runtime type → GPU**. Cell đầu kiểm tra CUDA và in tên GPU.
3. Upload ZIP vào đúng folder Drive ghi trong cell cấu hình; đợi upload hoàn tất.
4. Chạy cell cài thư viện, rồi cell mount Drive/cấu hình. Chỉ cấp quyền Drive cho notebook bạn kiểm soát.
5. Lần đầu giữ `FULL_TRAIN=False`, `RESUME=False`, `RUN_ID=''`: chạy thử 200 train / 80 validation, 2 epoch.
6. Chạy cell kiểm tra/giải nén, TensorBoard nếu cần và cell train. Kiểm tra model nhận `[4,3]` và checkpoint lưu được lên Drive.
7. Khi chạy thử đạt, đặt `FULL_TRAIN=True`, `RESUME=False`, `RUN_ID=''`. Chạy lại từ cell cấu hình và chuẩn bị dữ liệu trước khi train đầy đủ.
8. Sau train, chạy cell cuối để xuất `best.pt`; có thể xếp cell này chờ sau cell train. Xác nhận output báo lưu vào `exports`.

Notebook 10.000/1.500 dùng `IMGSZ=960`, `BATCH=4`, tối đa 100 epoch và `patience=20`. Tên folder Drive có tiền tố cũ trong notebook chỉ là đường dẫn của lượt đã chạy; repository không phụ thuộc dự án khác. Nếu đổi folder, cập nhật đường dẫn cho khớp ZIP/checkpoint thật.

Mỗi lượt mới cần run ID mới. Không chạy hai phiên ghi cùng run. Weights khởi tạo là YOLO11n-Pose pretrained 17 keypoints; Ultralytics đổi đầu pose thành 4 điểm theo dataset, nên thông báo override 17 → 4 là dự kiến.

## 9. Đọc chỉ số và resume

### Chỉ số

| Chỉ số | Ý nghĩa |
|---|---|
| Epoch | Lượt học toàn bộ tập train |
| Batch | Nhóm ảnh trong epoch; batch 4 với 10.000 ảnh tạo 2.500 nhóm |
| `pose_loss` | Loss vị trí bốn góc; theo dõi xu hướng giảm |
| `val/pose_loss` | Loss góc trên validation; nên giảm hoặc ổn định |
| `box_loss`, `dfl_loss` | Loss định vị khung bao |
| `cls_loss` | Loss phân loại đối tượng biển |
| `kobj_loss` | Loss độ tin cậy/hiển thị keypoint |
| Box `(B)` / Pose `(P)` | Chỉ số khung bao / điểm góc |
| Precision / Recall | Ít nhận nhầm / ít bỏ sót khi giá trị cao |
| Pose mAP50–95 | Đánh giá tổng hợp qua nhiều ngưỡng khớp keypoint |
| Checkpoint Drive: epoch N | Tiến độ đã lưu bền vững đến epoch N |

`0.995` là 99,5% điểm metric, **không có nghĩa 99,5% ảnh đúng tuyệt đối cả bốn góc**. Phải xem ảnh mới, nhất là biển chéo/nhỏ/chói/đỏ. Dao động một epoch chưa đủ kết luận lỗi; train loss giảm nhưng validation xấu kéo dài cần kiểm tra lại.

`EarlyStopping` dừng vì không cải thiện theo tiêu chí trainer trong số epoch patience, không phải mặc nhiên hết giờ GPU. Lượt đã chạy dừng epoch 81, chọn best epoch 61; đây là lịch sử một lượt, không phải cấu hình cố định.

### Colab ngắt/reset

1. Giữ notebook, ZIP và folder checkpoint trên Drive. File runtime local có thể mất sau reset.
2. Kết nối lại GPU, cài dependencies và mount Drive.
3. Đặt đúng `FULL_TRAIN` của run cũ, `RESUME=True`, `RUN_ID` cũ; giữ cấu hình khớp contract.
4. Chạy cell chuẩn bị dữ liệu, rồi cell train để restore checkpoint đã kiểm tra.
5. Xác nhận output `Restored epoch` và run ID trước khi tiếp tục.

Epoch đang chạy dở có thể phải chạy lại. Run completed/early stop không cần ép train lại đủ 100. Fine-tune dataset mới là lượt mới, cần kế hoạch dữ liệu cũ/mới và validation phù hợp; không nhầm với resume.

Checkpoint có slot `last-a.pt`/`last-b.pt`, `best-a.pt`/`best-b.pt` và metadata hash. Dùng restore của notebook, không chọn slot theo tên rồi bỏ metadata. Để test, ưu tiên `exports/best.pt`, không dùng weights pretrained ban đầu.

## 10. Đăng ký và test model mới

### Tải và đăng ký

Tải `best.pt` từ `runs/<RUN_ID>/exports` trên Drive, đặt ngoài checkout, rồi chạy:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/register_test_model.py `
  --weights D:/ANPR-Training-Data/training-output/best.pt `
  --root D:/ANPR-Training-Data/model-yolo-point `
  --version run-20261103 `
  --name "YOLO11 Point - run 20261103" `
  --best-epoch 61
```

Thay version, tên và best epoch bằng lượt thật. Tiện ích sao chép **weights**, không sao chép ảnh, vào registry và tạo metadata SHA-256. Không ghi đè version cũ. Chỉ dùng weights train tin cậy.

### Test trên web

1. Tab **Test model** → **Tải lại model** → chọn version.
2. **Chọn ảnh** JPEG/PNG/WebP tối đa 15 MB → **Chạy test**. Server giới hạn thêm 25 triệu pixel.
3. Hoặc mở **Test model** từ ảnh đang duyệt để test đúng ảnh đó.
4. Xem bốn điểm trên ảnh gốc và crop nắn; có thể nhận 0 hoặc nhiều biển.
5. So cùng ảnh giữa các version. Confidence là độ tin cậy phát hiện, không phải độ chính xác từng góc.

Ảnh tải lên là tạm thời; test không tự sửa nhãn/duyệt ảnh. Preview/crop dùng canvas để giữ CSP hiện tại. Worker nạp model trong tiến trình riêng mỗi request; tổng chờ gồm thời gian nạp môi trường, không chỉ số giây inference in trên kết quả.

## 11. Xử lý lỗi

| Triệu chứng | Kiểm tra/cách xử lý |
|---|---|
|401 / yêu cầu đăng nhập | Đúng credentials và phiên trình duyệt; không bỏ auth để chữa lỗi |
|403 / outside LAN / invalid origin | Dải AllowNetwork, Host bằng IP hợp lệ và cùng origin |
| Cổng 8767 đã dùng | Dừng đúng server cũ hoặc chọn cổng khác |
| Không thấy model | Đúng ModelRoot, có best.pt/model.json và đã Tải lại model |
| Checksum model sai | Kiểm tra weights bị đổi; đăng ký version mới thay vì bỏ kiểm tra |
| `operator torchvision::nms does not exist` | Dùng inference venv và cặp torch/torchvision tương thích |
| Không chạy được model | Đúng InferencePython, dependencies và model pose bốn điểm; xem log ngoài checkout |
| Đang test ảnh khác | Inference được tuần tự hóa; đợi yêu cầu trước xong |
| Quá 90 giây / ảnh quá lớn | Thử ảnh nhỏ hơn, kiểm tra CPU; giới hạn 15 MB / 25 triệu pixel |
| Preview/crop không hiện | Ctrl+F5, xem Console; bản hiện tại dùng ImageBitmap/canvas, không gán base64 vào img |
| Colab không thấy ZIP | Đợi upload đủ, đúng tài khoản Drive, tên file và DATASET_ZIP |
| ZIP sai checksum | Dùng đúng cặp snapshot/notebook, không bỏ assert |
| Resume sai contract | Kiểm tra run ID/chế độ/cấu hình/dataset; không ghi lẫn vào run cũ |
| Metric cao nhưng góc ảnh mới sai | Kiểm tra nhãn, leakage và độ đa dạng ảnh khó; không tự duyệt bằng confidence |

HTML Test model được đọc lại theo request; code Python backend thay đổi cần restart đúng server. Giao diện cũ có thể cần Ctrl+F5.

## 12. Tests và tài liệu

Cần Python có Pillow và Node.js cho test JavaScript. Test ảnh nâng cao cần dependencies tương ứng.

```powershell
python -m unittest discover -s tests/anpr-training
Get-ChildItem tests/anpr-training/test_*.cjs | ForEach-Object {
  node $_.FullName
  if ($LASTEXITCODE -ne 0) { throw "JavaScript test failed: $($_.Name)" }
}
```

Test mặc định dùng fixture tổng hợp. [Smoke train CPU](tests/anpr-training/test_colab_training_smoke.py) phải bật riêng; skip mặc định không phải lỗi train. Tests không tự train trả phí hoặc truy cập dữ liệu thực của người dùng.

- [Test model và registry](docs/anpr-model-testing.md).
- [Chuẩn bị dữ liệu và web duyệt ban đầu](docs/anpr-pose-training-preparation.md).
- [Mở rộng ACS/Roboflow và OCR](docs/anpr-vietnam-expansion.md).
- [Duyệt nhóm ô tô](docs/anpr-car-group-review.md).
- [Snapshot mixed v2](docs/anpr-pose-mixed-v2.md).
- [Snapshot mixed v3](docs/anpr-pose-mixed-v3.md).

Tài liệu snapshot lịch sử có số lượng/quy tắc riêng. Ưu tiên manifest và notebook khớp gói đang dùng, cùng README hiện tại cho entrypoint web.
