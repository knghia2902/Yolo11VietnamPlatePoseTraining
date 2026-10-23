# Duyệt nhóm ô tô trước khi chia bộ Pose v3

Owner yêu cầu ngày 2026-10-22: OCR để đề xuất nhóm ô tô, duyệt lại bằng mắt, sau đó chia train/validation đều có ô tô và xe máy. Đây là công cụ chuẩn bị dữ liệu độc lập; không thay ANPR runtime hoặc lấy thuật toán từ ACSv2.

OCR chạy cục bộ bằng `rapidocr_onnxruntime==1.4.4` trong môi trường Python 3.12 riêng. Dùng bốn góc đã duyệt để nắn biển số; chạy nhận dạng nguyên crop và phát hiện dòng chữ. Không tự đổi O/0, I/1 hoặc B/8. Confidence và dạng biển chỉ giúp chọn đề xuất, không chứng minh biển số đúng. Dữ liệu, biển số, ảnh cắt và lịch sử duyệt nằm ngoài Git.

## Cách duyệt

Mở trang LAN cổng 8766 bằng tài khoản trang duyệt hiện có. Nhập tên người duyệt, đối chiếu ảnh gốc và crop, sửa biển số nếu sai. Dấu chấm/gạch được bỏ khi lưu. Xem các ảnh cùng biển và các đề xuất gần giống để phát hiện OCR lệch một ký tự.

- **Lưu** ghi chỉnh sửa vào máy chủ, chưa xác nhận đúng.
- **Duyệt** ghi biển số, nhóm, người duyệt và chuyển ảnh tiếp theo. Enter thực hiện Duyệt khi con trỏ không nằm trong ô nhập.
- Phím trái/phải chuyển ảnh khi không nhập chữ. Chuyển ảnh sẽ lưu chỉnh sửa còn chờ; lỗi lưu giữ nguyên ảnh và hiện thông báo.
- Bộ lọc mặc định **Chưa duyệt**. Có thể xem tất cả hoặc đã duyệt để sửa lại.
- Cùng biển số luôn cùng nhóm. Mã nhóm bổ sung có thể nối thêm các ảnh cần ở chung phía train/val; mã khác nhau không tách được cùng biển.
- Biển không rõ: Lưu và giữ chưa duyệt, không đoán chữ để vượt điều kiện. Kiểu biển ngoài dạng phổ thông cần xử lý riêng trước khi đóng gói.
- Xuất bản duyệt tải JSON dự phòng; bản chính đã lưu trên máy chủ. Đóng trình duyệt không mất các lần lưu thành công. Nếu hai cửa sổ cùng sửa một ảnh, phiên lưu cũ bị chặn, cần tải lại để đối chiếu.

Workspace: `D:/nATime-quarantine/20261022-car-group-review`. Mỗi lần lưu có phiên bản, gắn SHA256 manifest nguồn; các phiên bản trước nằm trong `group-history/`. Tài khoản nằm ngoài Git, không nhúng trong URL. Server chỉ phục vụ đường dẫn ảnh đã liệt kê trong manifest, có Basic auth và giới hạn mạng LAN. Máy chủ phải còn bật để máy khác duyệt.

## Tái chạy

```powershell
& D:/nATime-quarantine/20261022-plate-group-ocr/venv/Scripts/python.exe scripts/test-tools/Yolo11VietnamPlatePoseTraining/car_group_review.py ocr --dataset D:/nATime-quarantine/20261022-anpr-pose-mixed-v2/dataset-mixed-v2 --workspace D:/nATime-quarantine/20261022-car-group-review
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/car_group_review.py serve --workspace D:/nATime-quarantine/20261022-car-group-review --credentials D:/nATime-Secrets/anpr-review-lan/credentials.json --bind 0.0.0.0 --port 8766 --allow-network 10.32.0.0/16
```

OCR tiếp tục từ các đề xuất đã ghi khi bị ngắt; không ghi đè bản duyệt. Server chặn tiến trình thứ hai trên cùng workspace. Không dùng đồng thời server ghi và lệnh lập kế hoạch; dừng server sau khi hoàn tất duyệt để chốt snapshot.

## Điều kiện bộ mới

Lệnh dưới dừng nếu bất kỳ ảnh ô tô nào chưa duyệt, thiếu biển số đúng dạng hoặc thiếu người duyệt:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/car_group_review.py plan --workspace D:/nATime-quarantine/20261022-car-group-review --out D:/nATime-quarantine/20261022-car-group-review/reviewed-split-plan.json
```

Kế hoạch nối cùng biển số đã duyệt, nhóm bổ sung, hash bytes/pixel giống nhau và mọi cạnh dHash trong ngưỡng nguồn. Giữ nguyên nhóm nguồn xe máy: green ở train, bv1/cam ở validation. Chọn khoảng 20% ảnh ô tô cho validation bằng các nhóm nguyên vẹn; yêu cầu cả hai loại xe có mặt ở cả hai split. Không ép tỷ lệ bằng cắt nhóm. Có xung đột liên thông giữa hai nguồn xe máy thì dừng để xử lý, không âm thầm bỏ cạnh.

Ngày 2026-10-23, owner yêu cầu bỏ 7 ảnh còn chưa duyệt. Dùng công cụ đóng gói [v3](anpr-pose-mixed-v3.md) với `--exclude-pending-count 7`: chốt snapshot khi server đã dừng, loại đúng các ảnh chờ theo số lượng đã được phép, giữ 844 ảnh ô tô đã duyệt. Lệnh `plan` cũ vẫn yêu cầu duyệt mọi ảnh và không tự loại ảnh. Bản v3 giữ ảnh gốc để khôi phục, không công bố danh sách loại hoặc OCR private lên GitHub.

Sau khi owner duyệt đủ, tạo gói v3 và notebook mới gắn manifest/ZIP riêng, giữ bốn góc và 47 ảnh đã loại; xác minh output rồi mới dùng. Vì ảnh ô tô validation mới từng có trong train v2, phải train lại từ pretrained cho lượt đánh giá này; không dùng `best.pt`/resume v2 để tuyên bố chất lượng trên validation mới. Kế hoạch chưa phải ZIP hoặc notebook v3 đã phát hành. Không có metadata đủ để chứng minh tuyệt đối không trùng xe giữa các nguồn; cần người duyệt đối chiếu biển số và nhóm, dHash chỉ bổ sung phát hiện gần trùng.
