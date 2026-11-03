# Test model YOLO11 Point trên web duyệt

Trang test dùng model pose đúng bốn điểm theo thứ tự TL → TR → BR → BL. Đây là tool kiểm tra dữ liệu và model cục bộ trong repository huấn luyện độc lập. OCR là luồng riêng. Kết quả test không cập nhật nhãn hoặc tự duyệt ảnh.

## Thành phần được lưu trong Git

- Source chuẩn bị dữ liệu, web duyệt, chia train/validation, tạo notebook và checkpoint.
- `pose_model_test_server.py`: web duyệt và trang `/model-test`, giữ Basic Auth và giới hạn LAN.
- `pose_model_test_worker.py`: inference CPU trong tiến trình riêng, timeout 90 giây.
- `register_test_model.py`: đăng ký một version weights tin cậy, không ghi đè version trước.
- Các bài kiểm tra dùng fixture tổng hợp và notebook không có output.

Ảnh, ZIP dataset, SQLite, manifest dữ liệu thực, weights, secrets và log không nằm trong repository. Người dùng phải cung cấp các tệp này từ vùng lưu trữ riêng. Các hướng dẫn v2/v3 mô tả snapshot lịch sử; số lượng ảnh và validation hiện tại phải lấy từ manifest của gói đang dùng, không suy ra từ các tài liệu cũ.

## Môi trường inference

Tạo virtual environment riêng. Pin `ultralytics==8.4.183`, `Pillow==11.3.0`, OpenCV và cặp PyTorch/torchvision tương thích. Cấu hình CPU đã kiểm tra: `torch==2.10.0`, `torchvision==0.25.0+cpu` từ wheel CPU chính thức của PyTorch. Không trộn torchvision CUDA với torch CPU. Web có thể chạy bằng interpreter khác; `--inference-python` phải trỏ vào môi trường inference này.

## Đăng ký model sau mỗi lần train

Chỉ dùng weights từ lượt train tin cậy của bạn; không nhận tệp `.pt` từ người lạ. Mỗi version có một folder chứa `best.pt` và `model.json`. Chạy từ root checkout:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/register_test_model.py `
  --weights D:/private/training-output/best.pt `
  --root D:/private/models `
  --version run-20261103 `
  --name "YOLO11 Point - run 20261103" `
  --best-epoch 61
```

Tiện ích sao chép weights vào registry riêng; không sao chép dataset. SHA-256 được kiểm tra trước inference. Worker xác nhận model pose có `kpt_shape: [4, 3]`. Bấm **Tải lại model** để thấy version mới.

## Chạy web

Sử dụng workspace đã được chuẩn bị bằng tool duyệt và tệp credentials riêng chứa `username`/`password`. Không commit credentials hoặc đặt chúng trong URL.

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_server.py `
  --workspace D:/private/expansion `
  --credentials D:/private/secrets/review.json `
  --model-root D:/private/models `
  --inference-python D:/private/inference-venv/Scripts/python.exe `
  --bind 0.0.0.0 --port 8767 --allow-network 10.32.0.0/16
```

Chỉ chạy một server trên cùng cổng. Nếu khởi động nền bằng PowerShell, dùng `Start-Process -WindowStyle Hidden` và đưa log ra ngoài checkout. Không dùng entrypoint `curate_acs_training_pool.py serve` nếu cần trang test model.

## Kiểm tra trên trình duyệt

1. Đăng nhập web LAN bằng Basic Auth hiện có. Chọn tab **Test model**.
2. Chọn version model, chọn ảnh JPEG/PNG/WebP tối đa 15 MB và bấm **Chạy test**. Hoặc mở **Test model** từ ảnh đang duyệt.
3. Kiểm tra bốn điểm và crop nắn. Confidence là điểm phát hiện biển, không phải độ chính xác từng góc.
4. So thêm ảnh chéo, chói, đêm, biển nhỏ và biển đỏ ngoài dữ liệu train/validation trước khi kết luận chất lượng.

Ảnh được giải mã bằng `createImageBitmap` và vẽ lên canvas. CSP vẫn giữ `img-src 'self'`; không gán `data:image` vào thẻ ảnh. Kiểm tra API thành công không thay thế kiểm tra upload, preview và crop trực tiếp trên trình duyệt.

Giao diện test dùng cùng style với thanh duyệt: Segoe UI 13px, control 28px, tab gạch chân, nút trung tính và bóng nổi/lõm nhẹ. Lần chỉnh style gần nhất đã được phục vụ qua HTTP nhưng còn chờ xác nhận trực quan của người dùng; không coi kiểm tra API là nghiệm thu thiết kế.

## Kiểm tra tự động

```powershell
python -m unittest discover -s tests/anpr-training
Get-ChildItem tests/anpr-training/test_*.cjs | ForEach-Object { node $_.FullName; if ($LASTEXITCODE -ne 0) { throw "Test failed" } }
```

Bài smoke train CPU được tắt mặc định; cần bật theo hướng dẫn trong test và chuẩn bị môi trường train riêng. Không có CI nào tự truy cập dữ liệu thực hoặc chạy train trả phí.
