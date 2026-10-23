# YOLO11 Pose v3: validation có ô tô và xe máy

Owner yêu cầu ngày 2026-10-23: bỏ 7 ảnh ô tô còn chưa duyệt, giữ 844 ảnh ô tô đã xác nhận biển số/nhóm, chia lại dữ liệu và phát hành notebook v3 lên repository `knghia2902/Yolo11VietnamPlatePoseTraining`. Không xóa vĩnh viễn ảnh gốc; danh sách loại và snapshot duyệt được lưu riêng ngoài Git, không nằm trong ZIP huấn luyện hoặc repository public.

| Split | Ô tô | Xe máy | Tổng |
|---|---:|---:|---:|
| Train | 675 | 1.748 | 2.423 |
| Validation | 169 | 1.472 | 1.641 |
| Tổng | 844 | 3.220 | 4.064 |

Có 533 nhóm biển số ô tô đã duyệt và 535 thành phần liên thông tổng cộng. Seed 42; chọn gần 20% ảnh ô tô cho validation bằng nhóm nguyên vẹn. Xe máy giữ nguyên nguồn: green ở train, bv1/cam ở validation. Vì giữ nhóm nguồn xe máy, tỷ lệ validation toàn bộ khoảng 40,38%, không ép về 20% bằng cách cắt nhóm.

## Kiểm soát rò rỉ và truy nguyên

Ô tô cùng biển số đã chuẩn hóa, cùng nhóm bổ sung hoặc nối qua ảnh trùng byte/pixel/dHash phải cùng split. Giữ nguyên nhóm và thành phần nguồn xe máy. Hợp các cạnh theo tính bắc cầu trước khi chia. Không tự duyệt OCR, không tạo nhóm riêng cho từng ảnh để đạt tỷ lệ đẹp. Loại đúng các ảnh chưa duyệt theo số lượng owner cho phép; nếu số lượng chờ thay đổi thì dừng. Ảnh ô tô được giữ phải có biển số đúng dạng và người duyệt.

Giữ nguyên bytes ảnh, nhãn và bốn góc đã duyệt. Manifest mới ghi hash nguồn, hash snapshot, loại xe, nhóm định danh băm và thành phần liên thông. Kiểm tra lại hash, nhãn, hình học, nhóm, ảnh trùng/gần trùng và tính đầy đủ của output sau khi copy. Tổng số ảnh đã loại là 54: 47 từ lượt duyệt góc trước và 7 từ lượt này. Dữ liệu private, OCR, bản duyệt, ZIP, weights và log không được đưa lên GitHub.

Đây là chống rò rỉ theo nhóm đã biết và bộ lọc dHash, không chứng minh tuyệt đối không có cùng xe ở các nguồn dưới góc rất khác. Cần kiểm định độc lập mới để kết luận khả năng tổng quát ngoài nguồn đã có. Các ảnh validation ô tô v3 từng thuộc train v2, vì vậy phải bắt đầu lại từ pretrained `yolo11n-pose.pt`; không resume hoặc dùng `best.pt` v2 để tuyên bố chất lượng v3. Chỉ số v2 và v3 dùng validation khác nhau nên không so trực tiếp.

## Tái tạo và các file phát hành

Dừng server duyệt nhóm trước khi đóng gói; tool giữ cùng khóa workspace trong toàn bộ thao tác để tránh sửa bản duyệt giữa chừng. Source vẫn là checkout chính, output ở ngoài Git:

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/package_reviewed_vehicle_v3.py --workspace D:/nATime-quarantine/20261022-car-group-review --out D:/nATime-quarantine/20261023-anpr-pose-mixed-v3/dataset-mixed-v3 --exclude-pending-count 7
```

Output phải chưa tồn tại. Thư mục cha chứa `dataset-mixed-v3-training.zip`, `Train_YOLO11n_Plate_Pose_Mixed_v3.ipynb`, `package-report-v3.json` và `review-snapshot.private.json`. Chỉ notebook, code, tài liệu và test được đưa lên repo. File báo cáo có hash ZIP/manifest/notebook để đối chiếu; không công bố snapshot hoặc dữ liệu thật.

## Chạy Colab

1. Mở notebook v3, chọn GPU rồi chạy lần lượt từ trên xuống.
2. Mount Drive. Dùng thư mục riêng `MyDrive/nATime-YOLO11-Pose-v3-reviewed-groups`; để `RESUME=False`, `RUN_ID=''` cho lượt mới.
3. Upload đúng `dataset-mixed-v3-training.zip`. Notebook pin SHA256 của ZIP, manifest và verifier; sau giải nén kiểm tra nhãn, rò rỉ và sự hiện diện của hai loại xe ở mỗi split.
4. Mặc định `FULL_TRAIN=True`, 100 epoch, imgsz 960, batch 4, patience 20; có thể đặt `FULL_TRAIN=False` để smoke 2 epoch trước.
5. Drive checkpoint sau mỗi epoch, dashboard và TensorBoard được giữ như v2. Resume chỉ cùng lượt v3, cùng manifest và cùng RUN_ID; có thể mất phần epoch đang chạy dở khi ngắt.
6. Sau train đầy đủ, chạy ô validation/export ONNX, rồi ô cuối đánh giá riêng ô tô/xe máy và hiển thị tối đa 6 ảnh dự đoán mỗi loại. Báo cáo riêng từng loại được lưu vào thư mục exports trên Drive.

V3 chưa được chạy train/GPU Colab trong task đóng gói. Kiểm thử local dùng dữ liệu tổng hợp, bao gồm chạy chính ô preflight notebook trên ZIP thật tổng hợp và xác nhận ZIP sai hash bị chặn. Không dùng legacy ACSv2 làm nguồn code/thuật toán; không thay runtime Gate, database, camera hoặc thiết bị.
