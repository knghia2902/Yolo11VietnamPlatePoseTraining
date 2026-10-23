# Chuẩn bị YOLO11 Pose bốn góc biển số

Phạm vi: công cụ dữ liệu offline theo yêu cầu ngày 2026-10-21. Không thay đổi pipeline Gate, không triển khai BV1, không tự tải ảnh lên dịch vụ ngoài.

## Phát hiện trên gói cũ

Gói ZIP cũ có 4.592 ảnh/nhãn. Toàn bộ nhãn lấy bốn góc bounding box song song trục ảnh, không phải ground truth bốn góc thật. Có 474 bản ảnh trùng byte. Cách chia ngẫu nhiên cũ làm 131/689 ảnh validation trùng byte với train. Vì vậy không dùng gói cũ để đánh giá khả năng nắn phối cảnh.

## Công cụ và quy trình

Chạy tại `D:\nATime`, Python có Pillow. Tuỳ chọn contour cần OpenCV và NumPy. Toàn bộ ảnh, review JSON, manifest, ZIP xuất phải ở ngoài checkout. Các đường dẫn đầu ra phải chưa tồn tại; tool không xoá hoặc ghi đè dữ liệu cũ.

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/prepare_pose_dataset.py scan --source scripts/test-tools/Yolo11VietnamPlatePoseTraining/vietnam_vehicles_pose_dataset.zip --out D:/nATime-quarantine/20261021-anpr-pose-review --propose-contours
```

Mở `review.html` trong thư mục đầu ra bằng trình duyệt. Trang chạy offline, không có thư viện hoặc endpoint mạng. Đường bao màu cam chưa được duyệt. Chọn lại bốn điểm theo thứ tự **TL → TR → BR → BL** trên ảnh gốc, nhập người duyệt, kiểm tra nhóm và bấm **Duyệt ảnh hiện tại**. Có thể giữ đề xuất nếu nó thật sự đúng. Loại ảnh không chắc chắn, ảnh che khuất hoặc ảnh nhiều biển chưa được gán đủ. Phiên bản tool này chỉ hỗ trợ một biển được gán đủ trên mỗi ảnh; muốn giữ ảnh nhiều biển phải dùng công cụ annotation nhiều đối tượng và mở rộng pipeline tương ứng.

Bấm **Tải reviews.json** để lưu tiến độ; trang không tự lưu khi đóng. Có thể nạp JSON trở lại để tiếp tục. Manifest giữ hash ảnh, tên gốc của tất cả bản trùng, nguồn và tình trạng kiểm duyệt. Không đưa ảnh hay manifest có tên nguồn vào Git.

Các ảnh cùng xe/lượt đi qua/chuỗi chụp phải có cùng `group`, kể cả khác camera. Mặc định dùng toàn bộ tiền tố nguồn làm nhóm bảo thủ; đây là chia theo nguồn, không phải tự suy ra session từ tên file. Chỉ tách nhỏ nhóm khi có metadata thật. Không đặt mỗi ảnh thành nhóm riêng. Nếu không biết xe hoặc session, giữ nguyên nhóm nguồn; tỷ lệ val có thể lệch nhiều hoặc không đủ nhóm độc lập, khi đó cần thêm tập kiểm định độc lập.

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/prepare_pose_dataset.py finalize --workspace D:/nATime-quarantine/20261021-anpr-pose-review --reviews D:/nATime-quarantine/20261021-anpr-pose-review/reviews.json --out D:/nATime-quarantine/20261021-anpr-pose-reviewed
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/prepare_pose_dataset.py verify --root D:/nATime-quarantine/20261021-anpr-pose-reviewed
```

`scan` bỏ ảnh trùng theo hash RGB đã giải mã, giữ provenance. `finalize` yêu cầu mọi ảnh giữ lại có người duyệt góc và nhóm, kiểm tra tứ giác lồi/thứ tự/phạm vi, tạo lại bbox từ góc thật. Nhãn cũ không được tự coi là đã duyệt. Chỉ sau khi mọi kiểm tra đạt mới xuất thư mục và ZIP `*-training.zip`.

Trước chia train/val, tool hợp các nhóm cùng session và các cặp dHash cách nhau tối đa 4 bit thành thành phần liên thông; mỗi thành phần ở nguyên một split. Thuật toán chọn tỷ lệ gần 15% nhất có thể mà không cắt nhóm. dHash chỉ là bộ lọc gần trùng: có thể gộp nhầm hoặc bỏ sót ảnh cùng xe dưới góc khác. Metadata nhóm và đánh giá trực quan vẫn cần thiết; không tuyên bố loại được mọi rò rỉ ngữ nghĩa. Không hạ ngưỡng chỉ để đạt tỷ lệ đẹp. Test cuối xác minh không trùng RGB, nhóm hoặc dHash qua hai split.

`draft-split.json` ghi phân chia dự kiến từ nhóm nguồn ban đầu, có `training_ready=false`. Lần chạy thực tế ngày 2026-10-21: 4.118 ảnh sau dedup, 1.399 đề xuất contour, 2.719 fallback cần gán lại; draft 3.254 train / 864 val với ba thành phần liên thông. Chưa ảnh nào được người duyệt xác nhận. Có thể chạy lệnh `plan --workspace <thư-mục-review>` để tính lại preview; kết quả cuối lấy từ `finalize` sau duyệt.

## Gán tự động có kiểm tra lại

`--propose-contours` dùng vùng bbox cũ làm ROI, Canny và contour tứ giác để đề xuất bốn góc. Cách này hữu ích khi viền biển rõ, nhưng có thể chọn khung xe, viền trong hoặc nét chữ; không có confidence đã hiệu chuẩn và không tự duyệt. Khi không tìm được contour, tool giữ hình chữ nhật cũ để người duyệt thay thế, có nhãn `legacy_bbox_unreviewed`.

Cách mở rộng sau đó: gán một tập sạch có đủ ngày/đêm, xe máy/ô tô, làn và góc nghiêng; train model pose đầu tiên; dùng model đó đề xuất cho tập còn lại rồi kiểm tra lại. Không tự dùng dự đoán model làm ground truth của validation. Giữ validation có nhãn người duyệt và nhóm độc lập. Các điểm bị khuất không được phỏng đoán rồi đánh dấu nhìn thấy.

## Notebook

Notebook mới chỉ nhận gói đã duyệt, dùng verifier cùng phiên bản có SHA-256 cố định, không tự chia lại dữ liệu. Nó kiểm tra gói trước train, hiển thị tỷ lệ split thực tế, mặc định chạy smoke 2 epoch; chạy đầy đủ cần chủ động bật biến. Tắt flip ngang/dọc để không đảo thứ tự góc và chữ. Tắt mosaic/mixup/cutmix, ghi nhận môi trường, và export từ `best.pt` của lần train đầy đủ. Mặc định `imgsz=960`, batch 4; kiểm tra kích thước biển thực và VRAM trước khi chọn 640 hoặc batch lớn hơn. Không bảo đảm độ chính xác hay tốc độ khi chưa đo.

Chưa chạy Colab/GPU trong task này. Gói ZIP thô được giữ nguyên để truy nguyên. ACSv2 không được dùng làm nguồn thuật toán gán nhãn; đây là tooling mới độc lập, chưa phải nghiệm thu ANPR runtime hay tuyên bố tương đương chức năng Gate.

Kiểm tra local: 14 test Python trên dữ liệu tổng hợp (bao gồm xuất ZIP rồi chạy chính cell preflight notebook), test logic JavaScript của trang duyệt, kiểm tra UTF-8/cú pháp và cấu hình train trên Ultralytics local 8.4.8 đều đạt. Notebook pin Ultralytics 8.4.183; chưa chạy môi trường pin trên Colab. Chưa kiểm tra giao diện bằng trình duyệt vì công cụ browser chặn URL `file://`. GitNexus index nATime commit `481f7df` chưa có bộ training mới; impact trả UNKNOWN, text search bổ sung không thấy caller runtime của hàm sinh mẫu cũ và hàm đó được giữ nguyên. Detect toàn checkout báo CRITICAL do tập thay đổi đang có (lần cuối tăng giới hạn trả về: 102 file/226 symbol/523 process), không phải kết quả sạch riêng cho task này; không commit/push hay reindex toàn repo trong task.

Tham khảo chính thức: [định dạng Pose](https://docs.ultralytics.com/datasets/pose/), [training](https://docs.ultralytics.com/modes/train/), [export](https://docs.ultralytics.com/modes/export/).

## Notebook có Google Drive checkpoint và theo dõi — 2026-10-22

Notebook hiện có ô mount Drive và cấu hình `FULL_TRAIN`, `RESUME`, `RUN_ID`, `IMGSZ`, `BATCH` trước khi nạp dữ liệu. Lần đầu upload ZIP đã duyệt; sau khi verify, ZIP được cache ở `MyDrive/nATime-YOLO11-Pose/datasets/dataset-training.zip`. Lần reconnect dùng cache và giải nén local. Dataset ZIP phát hành không thay đổi.

Mỗi run dùng thư mục `MyDrive/nATime-YOLO11-Pose/runs/<RUN_ID>/`. Sau mỗi epoch có model save, callback copy checkpoint còn optimizer/scaler/EMA sang hai slot luân phiên `last-a.pt`/`last-b.pt` và `best-a.pt`/`best-b.pt`, xác minh hash rồi commit `checkpoint.json`. Slot cũ được giữ trong khi cập nhật slot mới; nếu slot hiện tại lỗi, restore thử slot trước và sửa marker trước khi backup tiếp. Không cam kết giữ batch/epoch đang chạy dở hoặc tiến độ trước checkpoint đầu tiên. Drive lỗi sẽ dừng lượt train và báo, không báo giả đã lưu. Không chạy nhiều Colab vào cùng RUN_ID. Ghi/rename trên Drive phụ thuộc mount/FUSE và kết nối; đây không phải bảo đảm chống mọi sự cố dịch vụ cloud.

Resume: mở lại notebook mới, cài thư viện, mount cùng Drive, đặt cùng `FULL_TRAIN`, `RESUME=True` và `RUN_ID` cũ, chạy lại data preflight, mã hỗ trợ và ô train. Kiểm tra contract dataset/config/version trước phục hồi. Pin Ultralytics8.4.183 hỗ trợ cập nhật đường dẫn YAML và `save_dir` khi resume; tiếp tục optimizer/scheduler thay vì khởi tạo model mới. Có thể giảm BATCH khi thiếu VRAM. Run đã hoàn tất sẽ bỏ train để export. ONNX/best.pt/báo cáo export cũng được lưu vào `exports/` trên Drive.

Theo dõi: chạy ô TensorBoard trước ô train và xem Scalars khi train đang chạy. Bảng HTML trong ô train cập nhật khoảng20giây/lần, hiển thị phase(training/validating/saving_checkpoint/epoch_saved/completed), epoch, batch, loss, metrics, thời gian epoch và epoch checkpoint đã commit. Log mặc định Ultralytics hiển thị GPU memory. `status.json` trên Drive ghi thời gian UTC hoạt động gần nhất; giá trị cũ không chứng minh runtime còn sống hoặc đã chết. CSV/TensorBoard logs được backup mỗi epoch và phục hồi khi resume. Không có công cụ của task tự đăng nhập Google hoặc upload dữ liệu; người dùng thực hiện mount/cấp quyền trong Colab.

Kiểm thử:29 test nhẹ đạt;1 CPU integration riêng với Ultralytics8.4.183 trên4 ảnh train/2val tổng hợp đã ngắt sauepoch1 và resumeepoch2 thành công, xác nhận optimizer trong checkpoint Drive không bị strip, data path/save_dir mới áp dụng. Test dùng filesystem local mô phỏng Drive, chưa chạy mount Drive/TensorBoard widget/GPU Colab thật. Runtime test dependencies nằm ngoài Git; không thay package Python global. Báo cáo: `D:/nATime-quarantine/20261022-colab-checkpoint-tests/synthetic-1e30291b/smoke-report.json`.

## Duyệt qua LAN và lưu trên máy chủ

Owner chọn truy cập trong LAN ngày 2026-10-21. Máy chủ hiện tại có Ethernet `10.32.0.180` và Wi-Fi `10.32.8.171`; mở `http://10.32.0.180:8765/` hoặc `http://10.32.8.171:8765/` từ máy cùng mạng `10.32.0.0/16`. Không xuất bản Internet. HTTP này chỉ dùng trong LAN tin cậy.

Đăng nhập bằng tài khoản trong `D:\nATime-Secrets\anpr-review-lan\credentials.json`; file đã giới hạn quyền cho tài khoản Windows hiện tại, Administrators và SYSTEM. Không đưa mật khẩu vào URL/Git/log. Địa chỉ LAN có thể thay đổi theo DHCP; máy chủ phải bật và tiến trình Python phải chạy khi duyệt.

Tool `serve_pose_review.py` phục vụ riêng trang duyệt và danh sách ảnh đã biết; các đường dẫn manifest, file lưu và secret không được phục vụ. Client ngoài subnet bị chặn. Mỗi lần chỉnh điểm/nhóm/tên người duyệt/loại ảnh tự lưu sau 600 ms. Nút **Lưu ngay**, **Duyệt ảnh hiện tại** và điều hướng đều gửi lưu; chờ dòng **Đã lưu vào máy chủ** trước khi đóng trang. Mất kết nối hoặc xung đột phiên bản sẽ báo **CHƯA LƯU**, chặn điều hướng; tải JSON giữ bản local trước khi tải lại.

Theo yêu cầu owner ngày 2026-10-21, **Duyệt ảnh hiện tại** đồng thời xác nhận **Đã kiểm tra nhóm nguồn/lượt xe**; không cần tích riêng trước khi duyệt. Vẫn yêu cầu đủ bốn điểm, nhóm và tên người duyệt. Đây là xác nhận của người bấm duyệt, không phải tool tự suy ra nhóm đúng.

Cập nhật giao diện tiếp theo trong ngày: nút **Duyệt** lưu xác nhận góc/nhóm rồi tự chuyển sang ảnh tiếp theo chỉ sau khi server báo lưu thành công; vị trí khôi phục cũng trỏ tới ảnh tiếp theo. Chặn bấm đúp trong lúc lưu. Nút **Lưu** chỉ lưu ảnh đang xem. **Loại** là nút trên hàng thao tác; bỏ hai checkbox loại/kiểm tra nhóm, ẩn file import trên LAN, bỏ nhãn “Nạp bản duyệt” và dòng hướng dẫn tự lưu. Autosave, trạng thái đang lưu/đã lưu/lỗi và chức năng tải bản sao vẫn hoạt động.

Ô chọn cách đặt góc có ba chế độ: **Chọn điểm** (bấm TL → TR → BR → BL), **Kéo hình chữ nhật** và **Kéo hình vuông** (giữ chuột tại một góc, kéo tới góc đối diện rồi thả). Khi kéo có đường bao xem trước; chỉ khi thả chuột mới thay nhãn, bỏ trạng thái đã duyệt và tự lưu. Hình vuông được tính bằng pixel ảnh gốc, không bằng tọa độ chuẩn hóa. Có thể kéo theo cả bốn hướng, khung được giới hạn trong ảnh. Chuyển về Chọn điểm để đặt lại bốn góc thật với biển nghiêng; hai chế độ kéo tạo khung song song trục ảnh.

Phím tắt bổ sung ngày 2026-10-22: **Enter** duyệt và sang ảnh tiếp theo; **←** ảnh trước; **→** ảnh sau. Khi con trỏ ở ô nhập/select hoặc đang gõ tiếng Việt, phím giữ chức năng nhập liệu. Không lặp thao tác khi giữ phím, đang kéo khung hoặc đang chờ lưu.

Chọn **Lọc → Chưa duyệt** để chỉ xem ảnh chưa được xác nhận. Trước/Sau, phím mũi tên và nút Duyệt đều đi trong tập ảnh đang lọc, bỏ qua ảnh đã loại. Có thêm **Tất cả** và **Đã duyệt** để đối soát lại. Duyệt ảnh cuối sẽ báo **Không còn ảnh chưa duyệt**; đổi lại bộ lọc để tiếp tục xem. Chuyển bộ lọc luôn lưu ảnh đang chỉnh trước; nếu lưu lỗi/xung đột thì giữ nguyên bộ lọc và bản chỉnh hiện tại.

**Loại ảnh này** trên bản LAN lưu ngay, chuyển file ảnh từ `images/` sang `excluded-images/` trong workspace quarantine, ẩn khỏi endpoint ảnh và chuyển sang ảnh tiếp theo. Ảnh đã loại bị bỏ qua khi điều hướng/mở lại trang. Có thể phục hồi file từ quarantine nếu loại nhầm; đây không phải xóa vĩnh viễn. ZIP nguồn giữ nguyên. Bản offline chỉ có thể lưu cờ loại vào JSON, không tự di chuyển file trên đĩa.

Tiến độ nằm tại `D:\nATime-quarantine\20261021-anpr-pose-review\saved-reviews\`: một JSON cho mỗi ảnh đã thay đổi, ghi bằng file tạm, flush/fsync rồi thay thế nguyên tử. Có lịch sử phiên bản trước trong `history/`. Chỉ một tiến trình server được phép giữ workspace; nhiều trình duyệt dùng optimistic version trên mỗi ảnh để chống ghi đè. Đây là file trạng thái của tool, không thay đổi schema hay CSDL sản phẩm. Sau restart, server nạp lại chỉnh sửa và ảnh được lưu cuối. Có thể dùng **Tải reviews.json** hoặc endpoint đăng nhập `/api/export` lấy bộ nhãn đầy đủ cho lệnh `finalize`.

Nếu đang duyệt bản offline cũ, tải `reviews.json` trước khi đóng để giữ tiến độ. Bản online không tự nhập hàng loạt nhãn offline lên server, nhằm tránh ghi đè công việc trên máy khác. Giữ JSON đó để đối soát trước khi tiếp tục nếu đã có nhãn duyệt offline.

Khởi động lại thủ công từ checkout (không tự khởi động khi reboot):

```powershell
python scripts/test-tools/Yolo11VietnamPlatePoseTraining/serve_pose_review.py --workspace D:/nATime-quarantine/20261021-anpr-pose-review --credentials D:/nATime-Secrets/anpr-review-lan/credentials.json --bind 0.0.0.0 --port 8765 --allow-network 10.32.0.0/16
```

Nếu máy khác không truy cập được do Windows Firewall, chạy PowerShell **Run as Administrator**:

```powershell
& 'D:\nATime\scripts\test-tools\Yolo11VietnamPlatePoseTraining\Enable-LanReviewFirewall.ps1'
```

Rule chỉ cho Python TCP 8765, hai địa chỉ LAN hiện tại và client `10.32.0.0/16`. Phiên Codex hiện tại không có quyền Administrator nên chưa thêm rule này; kiểm tra HTTP trên chính máy chủ đạt với cả hai địa chỉ LAN, chưa xác nhận kết nối từ máy vật lý thứ hai. Không tự đổi policy firewall của Domain hay cấu hình router.
