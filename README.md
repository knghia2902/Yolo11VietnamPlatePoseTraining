# YOLO11 Pose — bốn góc biển số Việt Nam

Công cụ chuẩn bị dữ liệu, duyệt nhãn và huấn luyện YOLO11n-Pose với thứ tự góc **TL → TR → BR → BL**, `kpt_shape: [4, 3]`. Mỗi ảnh hiện hỗ trợ một biển số. Đây là repository độc lập để chuẩn bị dữ liệu, huấn luyện và kiểm tra model biển số.

## Test model sau mỗi lần train

### Source web nằm ở đây

| Thành phần | File |
|---|---|
| Khởi chạy web từ root | [Start-Web.ps1](Start-Web.ps1) |
| Backend web duyệt + test model | [pose_model_test_server.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_server.py) |
| Giao diện Train YOLO point / OCR | [review_expansion.html](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_expansion.html) |
| Giao diện Test model | [review_model_test.html](scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_model_test.html) |
| Worker chạy model và nắn biển | [pose_model_test_worker.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_worker.py) |
| Đăng ký model mới | [register_test_model.py](scripts/test-tools/Yolo11VietnamPlatePoseTraining/register_test_model.py) |
| Notebook 10.000 train / 1.500 validation | [Train_YOLO11_Point_10000.ipynb](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11_Point_10000.ipynb) |

Đây là web Python có backend và dữ liệu riêng, không phải website tĩnh chạy bằng GitHub Pages. GitHub lưu đầy đủ source; chạy web bằng `Start-Web.ps1` trên máy chứa workspace/model. Xem [requirements-web.txt](requirements-web.txt), [requirements-inference.txt](requirements-inference.txt) và [hướng dẫn](docs/anpr-model-testing.md).

```powershell
.\Start-Web.ps1 -Workspace D:/private/expansion -Credentials D:/private/secrets/review.json `
  -ModelRoot D:/private/models -InferencePython D:/private/inference-venv/Scripts/python.exe
```

Muốn phục vụ trong LAN, bổ sung `-Bind 0.0.0.0 -AllowNetwork 10.32.0.0/16`. Giữ Basic Auth và firewall phù hợp.

Xem [hướng dẫn chạy web và đăng ký model](docs/anpr-model-testing.md). Web có các tab Train YOLO point, OCR biển số và Test model; hỗ trợ ảnh tải lên hoặc ảnh đang duyệt, hiển thị bốn góc và ảnh nắn. Dự đoán không tự ghi đè nhãn đã duyệt.

## Bắt đầu

1. Đọc [hướng dẫn chuẩn bị dữ liệu](docs/anpr-pose-training-preparation.md): loại trùng, đề xuất contour, duyệt bốn góc thật và nhóm xe, rồi xuất gói đã xác minh.
2. Dùng `serve_pose_review.py` để duyệt qua LAN và lưu trên máy chủ; `review_pose.html` là giao diện duyệt góc. Không dùng bốn góc bounding box cũ làm nhãn thật mà chưa kiểm tra.
3. Đọc [bộ mixed v2](docs/anpr-pose-mixed-v2.md) trước khi dùng notebook v2. Train v2 có ô tô và xe máy, validation v2 chỉ có xe máy.
4. Bộ v3 dùng [nhóm ô tô đã duyệt và validation hỗn hợp](docs/anpr-pose-mixed-v3.md). Đã loại 7 ảnh còn chưa duyệt theo yêu cầu owner; 4.064 ảnh còn lại có cả ô tô và xe máy trong mỗi split. `package_reviewed_vehicle_v3.py` yêu cầu đúng số ảnh chưa duyệt được phép loại, chốt snapshot có khóa, kiểm tra dữ liệu rồi tạo ZIP và notebook khớp hash. Đọc [hướng dẫn duyệt nhóm](docs/anpr-car-group-review.md) nếu chuẩn bị bộ dữ liệu khác.

Chạy script từ root repository. Chuẩn bị dữ liệu và kiểm thử nhẹ cần Python với Pillow; đề xuất contour cần OpenCV/NumPy. OCR nhóm ô tô dùng môi trường Python 3.12 riêng với `rapidocr_onnxruntime==1.4.4` và Pillow. Notebook tự cài phiên bản thư viện train đã pin trên Colab.

## Notebook Colab

- [Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb): bộ mới, validation có ô tô và xe máy; có đánh giá và ảnh dự đoán riêng từng loại.
- [Train_YOLO11n_Plate_Pose.ipynb](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose.ipynb): bộ source-heldout ban đầu.
- [Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb](scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb): cấu hình train mixed v2.

Upload notebook lên Colab, chọn GPU, mount Google Drive rồi cung cấp ZIP đã duyệt đúng phiên bản. Notebook kiểm tra hash, hình học nhãn và rò rỉ trước khi train. Mặc định chạy smoke; chỉ bật `FULL_TRAIN` khi cấu hình đã đúng. Có checkpoint Drive sau mỗi epoch, resume cùng `RUN_ID`, TensorBoard, trạng thái train, validation và export ONNX từ `best.pt`. Xem tài liệu để hiểu giới hạn khi Colab/Drive ngắt.

Các notebook được lưu không có output hay ảnh dữ liệu. ZIP dữ liệu, weights và kết quả train không được phát hành trong Git. Hash trong notebook ràng buộc gói đã duyệt tương ứng; nếu chuẩn bị dữ liệu mới, dùng builder để tạo notebook/gói khớp nhau, không bỏ kiểm tra hash.

## Chống rò rỉ

Giữ toàn bộ nhóm xe/nguồn/lượt chụp và thành phần ảnh trùng/gần trùng ở một phía. OCR chỉ đề xuất, không tự duyệt; biển số cùng xe phải được xác nhận bằng mắt. dHash không chứng minh tuyệt đối không trùng xe ở góc khác. Nếu dùng ô tô từng ở train v2 làm validation mới, train lại từ pretrained để đánh giá; không resume model đã học các ảnh đó.

## Kiểm thử

Từ root repository:

```powershell
python -m pip install Pillow==11.3.0
python -m unittest discover -s tests/anpr-training
Get-ChildItem tests/anpr-training/test_*.cjs | ForEach-Object { node $_.FullName; if ($LASTEXITCODE -ne 0) { throw "JavaScript test failed" } }
```

Các test mặc định dùng dữ liệu tổng hợp. Bài integration train CPU cần môi trường Ultralytics riêng và bật tùy chọn theo file test; mặc định được skip.

Không commit ảnh, biển OCR, manifest, review JSON, credentials, checkpoint, dataset ZIP, ONNX hoặc log vận hành. Lưu chúng ngoài checkout. Không upload dữ liệu của người dùng lên GitHub khi chạy quy trình này.

## Mở trực tiếp trên Colab

- [Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb)
- [Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb)
- [Train_YOLO11n_Plate_Pose.ipynb](https://colab.research.google.com/github/knghia2902/Yolo11VietnamPlatePoseTraining/blob/main/scripts/test-tools/Yolo11VietnamPlatePoseTraining/Train_YOLO11n_Plate_Pose.ipynb)
