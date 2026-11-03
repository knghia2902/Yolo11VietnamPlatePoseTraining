# YOLO11 Pose — bốn góc biển số Việt Nam

Công cụ chuẩn bị dữ liệu, duyệt nhãn và huấn luyện YOLO11n-Pose với thứ tự góc **TL → TR → BR → BL**, `kpt_shape: [4, 3]`. Mỗi ảnh hiện hỗ trợ một biển số. Đây là tooling huấn luyện độc lập, chưa tích hợp model vào Gate.

## Bắt đầu

Web kiểm tra model sau mỗi lần train: xem [hướng dẫn test model](../../../docs/anpr-model-testing.md). Mục này mô tả entrypoint hiện tại, registry version, môi trường inference và kiểm tra trình duyệt.

1. Đọc [hướng dẫn chuẩn bị dữ liệu](../../../docs/anpr-pose-training-preparation.md): loại trùng, đề xuất contour, duyệt bốn góc thật và nhóm xe, rồi xuất gói đã xác minh.
2. Dùng `serve_pose_review.py` để duyệt qua LAN và lưu trên máy chủ; `review_pose.html` là giao diện duyệt góc. Không dùng bốn góc bounding box cũ làm nhãn thật mà chưa kiểm tra.
3. Đọc [bộ mixed v2](../../../docs/anpr-pose-mixed-v2.md) trước khi dùng notebook v2. Train v2 có ô tô và xe máy, validation v2 chỉ có xe máy.
4. Bộ v3 dùng [nhóm ô tô đã duyệt và validation hỗn hợp](../../../docs/anpr-pose-mixed-v3.md). Đã loại 7 ảnh còn chưa duyệt theo yêu cầu owner; 4.064 ảnh còn lại có cả ô tô và xe máy trong mỗi split. `package_reviewed_vehicle_v3.py` yêu cầu đúng số ảnh chưa duyệt được phép loại, chốt snapshot có khóa, kiểm tra dữ liệu rồi tạo ZIP và notebook khớp hash. Đọc [hướng dẫn duyệt nhóm](../../../docs/anpr-car-group-review.md) nếu chuẩn bị bộ dữ liệu khác.

Chạy script từ root repository. Chuẩn bị dữ liệu và kiểm thử nhẹ cần Python với Pillow; đề xuất contour cần OpenCV/NumPy. OCR nhóm ô tô dùng môi trường Python 3.12 riêng với `rapidocr_onnxruntime==1.4.4` và Pillow. Notebook tự cài phiên bản thư viện train đã pin trên Colab.

## Notebook Colab

- [Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb](Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb): bộ mới, validation có ô tô và xe máy; có đánh giá và ảnh dự đoán riêng từng loại.
- [Train_YOLO11n_Plate_Pose.ipynb](Train_YOLO11n_Plate_Pose.ipynb): bộ source-heldout ban đầu.
- [Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb](Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb): cấu hình train mixed v2.

Upload notebook lên Colab, chọn GPU, mount Google Drive rồi cung cấp ZIP đã duyệt đúng phiên bản. Notebook kiểm tra hash, hình học nhãn và rò rỉ trước khi train. Mặc định chạy smoke; chỉ bật `FULL_TRAIN` khi cấu hình đã đúng. Có checkpoint Drive sau mỗi epoch, resume cùng `RUN_ID`, TensorBoard, trạng thái train, validation và export ONNX từ `best.pt`. Xem tài liệu để hiểu giới hạn khi Colab/Drive ngắt.

Các notebook được lưu không có output hay ảnh dữ liệu. ZIP dữ liệu, weights và kết quả train không được phát hành trong Git. Hash trong notebook ràng buộc gói đã duyệt tương ứng; nếu chuẩn bị dữ liệu mới, dùng builder để tạo notebook/gói khớp nhau, không bỏ kiểm tra hash.

## Chống rò rỉ

Giữ toàn bộ nhóm xe/nguồn/lượt chụp và thành phần ảnh trùng/gần trùng ở một phía. OCR chỉ đề xuất, không tự duyệt; biển số cùng xe phải được xác nhận bằng mắt. dHash không chứng minh tuyệt đối không trùng xe ở góc khác. Nếu dùng ô tô từng ở train v2 làm validation mới, train lại từ pretrained để đánh giá; không resume model đã học các ảnh đó.

## Test các model sau mỗi lần train

Chạy web bằng `pose_model_test_server.py` với các tham số workspace/credentials/bind/port/allow-network như server duyệt, bổ sung `--model-root` (registry ngoài Git) và `--inference-python` (Python có Ultralytics/PyTorch tương thích). Trang `/model-test` cho chọn model, tải ảnh lên hoặc dùng nút **Test model** trên ảnh đang duyệt. Không lưu dự đoán vào nhãn và không tự duyệt ảnh.

Đăng ký weights tin cậy từ một lượt train mới bằng `register_test_model.py --weights <best.pt> --version <run-id> --name <tên-model> --best-epoch <epoch>`, rồi bấm **Tải lại model**. Registry mặc định là `D:/ANPR-Training-Data/model-yolo-point`. Tiện ích sao chép riêng weights nhỏ vào registry, không sao chép ảnh; từ chối ghi đè version cũ. Worker kiểm tra model phải có đúng bốn điểm trước inference. Không nhận weights tải lên từ trình duyệt. Test trên CPU nên lần gọi đầu có thêm thời gian nạp model.

### Kiểm thử tự động

Từ root repository:

```powershell
python -m pip install Pillow==11.3.0
python -m unittest discover -s tests/anpr-training
Get-ChildItem tests/anpr-training/test_*.cjs | ForEach-Object { node $_.FullName; if ($LASTEXITCODE -ne 0) { throw "JavaScript test failed" } }
```

Các test mặc định dùng dữ liệu tổng hợp. Bài integration train CPU cần môi trường Ultralytics riêng và bật tùy chọn theo file test; mặc định được skip.

Không commit ảnh, biển OCR, manifest, review JSON, credentials, checkpoint, dataset ZIP, ONNX hoặc log vận hành. Lưu chúng ngoài checkout. Không upload dữ liệu của người dùng lên GitHub khi chạy quy trình này.
