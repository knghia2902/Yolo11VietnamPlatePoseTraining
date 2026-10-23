# Bộ YOLO11 Pose v2: train ô tô và xe máy

Theo yêu cầu owner ngày 2026-10-22, tạo bộ train mới từ 4.071 ảnh đã duyệt. Không thay góc, không khôi phục 47 ảnh đã loại, không thay đổi hay ngắt lượt Colab v1.

## Phân chia và giới hạn đánh giá

| Split | Nguồn | Số ảnh |
|---|---|---:|
| Train | car | 851 |
| Train | green | 1.748 |
| Validation | bv1 | 1.466 |
| Validation | cam | 6 |
| Tổng | | 4.071 |

Train 2.599, validation 1.472 (36,16%). Tỷ lệ này là hệ quả giữ nguyên nhóm nguồn, không ép 80/20 bằng cách cắt nhóm. Train có cả ô tô và xe máy. Validation chỉ có xe máy, chưa phải kiểm thử ô tô độc lập. Không so trực tiếp mAP của v2 với v1 vì tập validation đã thay đổi. 851 ảnh car từng được dùng chọn model v1 nay vào train; không được dùng chúng làm test độc lập cho v2.

Chưa có metadata xe/lượt xe đủ tin cậy để tách nhỏ các nguồn. Không tạo vehicle ID từ tên file hoặc OCR chưa xác minh. Giữ nguyên nhóm người duyệt, nhóm nguồn/alias và thành phần liên thông cũ; nối thêm mọi cạnh dHash trong ngưỡng cũ. Nếu yêu cầu cắt qua thành phần liên thông, lệnh dừng. Chống rò rỉ ở mức nhóm đã biết và bộ lọc gần trùng; không chứng minh không có cùng xe dưới góc khác chưa được nhận diện ở hai nguồn. Cần metadata/kiểm duyệt vehicle ID nếu muốn train và validation đều có ô tô và xe máy với tỷ lệ nhỏ hơn.

## Tái tạo

Từ checkout chính, với Python/Pillow và bộ đã duyệt nằm ngoài Git:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/package_mixed_source_holdout.py --source D:/nATime-quarantine/20261022-anpr-pose-completed/dataset --out D:/nATime-quarantine/20261022-anpr-pose-mixed-v2/dataset-mixed-v2
```

Output phải chưa tồn tại. Lệnh xác minh input, tạo lại bố trí split, giữ nguyên bytes ảnh/nhãn, xác minh output và đóng ZIP, tạo notebook pin ZIP/manifest cùng báo cáo. Không sửa hàm `split_groups`, `finalize`, `verify` hoặc builder v1. ACSv2 không được dùng làm nguồn thuật toán; đây là tooling dữ liệu độc lập, không phải capability ANPR runtime mới.

## Chạy Colab

1. Upload `Train_YOLO11n_Plate_Pose_Mixed_v2.ipynb` lên Colab và bật GPU T4.
2. Chạy ô cài đặt, rồi ô Drive. Mặc định `FULL_TRAIN=True`, `RESUME=False`, `RUN_ID=''`.
3. Cấp quyền Drive, lưu RUN_ID mới; upload `dataset-mixed-v2-training.zip` ở ô nạp dữ liệu. Không upload ZIP v1 và không upload ZIP bọc cả kit.
4. Chờ verify 4.071 ảnh, xem bảng nguồn/split và overlay. Chạy TensorBoard, rồi ô train.
5. Mã checkpoint nhúng trực tiếp trong ô train để tránh lỗi bỏ sót helper. Checkpoint/log/export vẫn lưu Drive; resume theo cùng notebook v2/RUN_ID mới sau khi ngắt.

Thư mục Drive riêng: `MyDrive/nATime-YOLO11-Pose-v2-sourceholdout`. Cache ZIP v2 và hash pin ngăn dùng nhầm bộ v1. Không resume checkpoint v1 vì manifest đã đổi. Train bắt đầu từ pretrained YOLO11n Pose, giữ ảnh 960, batch 4, tối đa 100 epoch và early stopping 20 epoch. Không có upload hay đổi notebook Colab đang chạy tự động từ lệnh đóng gói.
