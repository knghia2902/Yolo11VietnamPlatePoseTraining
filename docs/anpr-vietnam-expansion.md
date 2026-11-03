# Mở rộng dữ liệu Việt Nam, giữ validation v3

Quyết định ngày 23/10/2026: chỉ thu thập biển Việt Nam, hướng tới 50.000 ảnh
train hữu ích. Giữ nguyên 1.641 ảnh validation v3, nhãn và tập thành viên của
chúng. Nền train hiện tại là 2.423 ảnh. Không đưa cả 4.064 ảnh v3 vào train.

## Duyệt dữ liệu

Chạy `curate_acs_training_pool.py` trong thư mục script training. Toàn bộ workspace
phải ở ngoài checkout. Các lệnh `init`, `inventory`, `import`, `process`, `ocr`,
`serve`, `status` đều nhận `--workspace` **trước** tên lệnh. `--help` mô tả tham số.
Model phải là model biển số đã train với `kpt_shape: [4, 3]`, không dùng model
pose người để suy ra góc biển. OCR sử dụng RapidOCR 1.4.4 trong Python 3.12.

Ví dụ khởi tạo (thay đường dẫn của máy):

```powershell
python curate_acs_training_pool.py --workspace D:/private/expansion init --dataset D:/private/dataset-mixed-v3 --snapshot D:/private/review-snapshot.private.json
python curate_acs_training_pool.py --workspace D:/private/expansion inventory --root '//server/share/Data/2026/1'
python curate_acs_training_pool.py --workspace D:/private/expansion import --archive D:/private/roboflow.zip --source-url https://universe.roboflow.com/workspace/project/dataset/1 --license 'CC BY 4.0'
python curate_acs_training_pool.py --workspace D:/private/expansion process --model D:/private/plate-best.pt
python curate_acs_training_pool.py --workspace D:/private/expansion ocr-all --follow
python expansion_triage.py --workspace D:/private/expansion --follow
python curate_acs_training_pool.py --workspace D:/private/expansion serve --credentials D:/private/credentials.json --bind 0.0.0.0 --port 8767 --allow-network 10.32.0.0/16
```

Inference và OCR có thể dùng hai interpreter khác nhau, cùng workspace SQLite.
Chỉ chạy một worker inference và một worker OCR. Không chạy lại launcher khi
worker cũ còn hoạt động. Dữ liệu xử lý theo từng ảnh, SQLite lưu giao dịch và lịch
sử phiên bản; chạy lại lệnh sau gián đoạn sẽ tiếp tục từ ảnh chưa xử lý. Không
xóa hay sửa ảnh ACS gốc trên máy chia sẻ. Chỉ máy chủ phục vụ preview/crop cục bộ.

1. Chọn **Validation: xác nhận biển số**. Nhập tên người duyệt, so ảnh với OCR,
   sửa biển số, đánh dấu đã kiểm tra, rồi bấm **Duyệt** hoặc Enter. Bốn góc của
   validation bị khóa. 169 ảnh ô tô đã có định danh duyệt trước được giữ lại;
   1.472 ảnh xe máy cần xác nhận. OCR không tự chuyển thành định danh tin cậy.
2. Chọn **ACS tháng 1** hoặc **Roboflow Việt Nam**. Kiểm tra đây là biển Việt Nam,
   có biển thật và thấy đủ bốn góc. Phóng ảnh 200–400%, kéo điểm theo thứ tự trên
   trái → trên phải → dưới phải → dưới trái. Chế độ chữ nhật/vuông chỉ giúp vẽ;
   ảnh nghiêng cần sửa lại từng điểm theo biên thật.
3. Kiểm tra biển số và ảnh gần trùng. So ảnh tham chiếu bên phải trước khi đánh
   dấu đã so. Bấm **Duyệt** để lưu và sang ảnh khác; **Lưu** giữ bản nháp. Phím
   trái/phải chuyển ảnh, có lưu bản nháp trước; nếu lỗi/đụng phiên bản thì giữ ảnh
   hiện tại và báo lỗi. Khi nhập text, phím tắt điều hướng không chạy.
4. Dùng **Loại** cho ảnh không có biển, biển nước ngoài, mờ hoặc cắt mất góc.
   Ảnh crop ACS nhỏ chỉ phục vụ OCR. Model có thể báo nhiều đề xuất giả: chỉ
   duyệt nếu đã kiểm tra toàn ảnh có đúng một biển Việt Nam và đủ bốn góc. Khi
   nhãn nguồn ghi nhiều biển, cần công cụ nhiều đối tượng; không duyệt thành nhãn
   chỉ một biển. Ảnh trùng chính xác được chặn.
5. **Xuất bản duyệt** tải JSON để dự phòng; nhãn đã lưu nằm trên máy chủ nên có
   thể tiếp tục từ máy LAN khác. Đây là bản duyệt, chưa phải ZIP train.

## OCR đầy đủ và tự đối chiếu với validation

Lệnh `ocr-all` tiếp tục đến hết hàng đợi duy nhất, bỏ qua ảnh trùng chính xác để
không OCR lặp. Lượt OCR schema 2 đọc phát hiện dòng và lượt nhận dạng thứ hai:
crop biển một dòng được đọc nguyên crop, biển hai dòng được tách để đọc riêng
từng dòng rồi ghép lại. Cờ `layout_aware` phân biệt lượt đọc phù hợp bố cục này
với các kết quả trước đó, để chạy tiếp không bỏ sót ảnh cần đọc lại.
ảnh nhiều vùng nghi biển được đọc từng vùng (tối đa tám, nhiều hơn phải duyệt).
Ảnh chưa có góc nhưng có hộp nguồn được OCR từ hộp; ảnh còn lại thử OCR preview
toàn cảnh. Cả hai cách dự phòng chỉ hỗ trợ OCR, không tạo nhãn bốn góc thật.

Việc đối chiếu dùng tất cả ứng viên biển số validation đúng định dạng, bao gồm
ứng viên chưa được người duyệt xác nhận. Biển đã được người sửa/xác nhận có ưu
tiên hơn OCR. Không cần chờ duyệt toàn bộ 1.472 định danh xe máy mới bắt đầu tự
đối chiếu. Mỗi kết quả giữ nguồn, điểm OCR, vùng đọc và ảnh tham chiếu.

Trên trang, chọn nguồn ACS/Roboflow rồi chọn bộ lọc:

- **Nghi trùng validation**: khớp chuỗi hoặc lệch/thừa/thiếu một ký tự so với
  ứng viên validation, hoặc dHash ảnh cách validation không quá bốn bit. Đối
  chiếu hash dùng toàn bộ validation, kể cả khi biển chưa OCR được. Xem ảnh
  tham chiếu bên phải; không tự sửa O/0 hay I/1.
- **OCR không chắc / chưa đọc**: định dạng không hợp lệ, điểm thấp hơn 0,90,
  hai cách đọc mâu thuẫn trên cùng vùng, OCR toàn cảnh/nhiều vùng hoặc lỗi đọc.
- **OCR rõ — chưa thấy trùng**: kết quả OCR qua các điều kiện phân nhóm, chưa
  tìm thấy trùng qua OCR. Hai cách đọc phải cùng cho một biển số với điểm từ
  0,90 trở lên (hoặc định danh đã được người xác nhận); vẫn cần duyệt góc và
  biển số trước khi train.
- **Chờ OCR đầy đủ**: đã xử lý ảnh nhưng chưa chạy xong lượt OCR mới.

Ngưỡng 0,90 là quy tắc phân hàng đợi, không phải bảo đảm xác suất đúng. Ảnh chưa
đọc được và ảnh validation chưa rõ vẫn cần người xử lý, không coi là không trùng.

Sau khi so ảnh và xác nhận biển, có thể chọn **Khác xe / khác biển số** cho kết
quả gần một ký tự, hoặc **Cùng xe — chặn khỏi train**. Đánh dấu đã so ảnh rồi
bấm **Lưu**. Không được dùng “Khác xe” để bỏ qua biển số khớp chính xác: nếu OCR
validation sai thì sửa định danh validation. Quyết định so ảnh được ràng buộc
với bộ tham chiếu; thêm hoặc đổi tham chiếu sẽ yêu cầu kiểm tra lại.

Worker `expansion_triage.py --follow` cập nhật các danh sách riêng dưới
`<workspace>/triage/` mỗi 30 giây, đến khi cả xử lý ảnh và OCR hoàn tất:
`summary.json`, `suspected_overlap.jsonl`, `confirmed_overlap.jsonl`,
`ocr_uncertain.jsonl`, `ocr_distinct.jsonl`, `ocr_pending.jsonl` và các danh sách
phụ. Đây là danh sách tham chiếu ID, không nhân bản ảnh hay đổi ảnh/nhãn validation.
Trang đọc trạng thái trực tiếp từ SQLite; bấm **Tải lại** để nhận kết quả mới.
Khi trang đang tải hoặc lưu, bộ chọn nguồn/lọc được khóa tạm để tránh nhầm dữ liệu.
Các hàng đợi cũng chứa kết quả đối chiếu của train v3 cũ (nguồn `baseline`), để
không bỏ qua rò rỉ tiềm ẩn ở phần dữ liệu nền. Nguồn ảnh/nhãn v3 được giữ nguyên;
trường hợp xác nhận trùng sẽ bị chặn khi lập bộ train mở rộng.

## Chống rò rỉ và giới hạn

Kiểm tra SHA-256 tệp, SHA-256 pixel RGB và đề xuất gần trùng bằng dHash khoảng
cách ≤4, bao gồm cả dữ liệu đã nhập. dHash gần chỉ là gợi ý, không tự kết luận
cùng xe. Biển số được chuẩn hóa dấu cách/ký tự phân cách; không tự thay O/0,
I/1 hoặc B/8. Các ảnh cùng biển số đã được người xác nhận phải ở cùng nhóm.

Khi người duyệt xác nhận một biển nằm trong validation, ảnh mới cùng biển bị
chặn động, kể cả nếu đã duyệt trước đó. Khi định danh validation chưa đủ, nhãn
mới có trạng thái **đã duyệt, chờ đối chiếu**, chưa đủ điều kiện đóng gói train.
Không được coi kiểm tra hash không trùng là bằng chứng không trùng xe. Ảnh có
biển không đọc được vẫn chưa đủ bằng chứng định danh; cần người xử lý tiếp.

Trước khi đóng gói bộ tiếp theo phải kiểm tra lại hash toàn bộ ảnh/nhãn validation,
định danh còn thiếu, nhóm xe, ảnh trùng và các ảnh nhiều biển/chưa duyệt. Huấn
luyện từ pretrained; model v1/v2 chỉ dùng đề xuất nhãn, không dùng để báo chất
lượng trên validation v3 vì các lượt train cũ có thể đã thấy ảnh holdout.

## Nguồn công khai

- [school, vietnamese-license-plate, v1](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0/dataset/1):
  8.357 ảnh export, CC BY 4.0, auto-orient và stretch 640×640, không augmentation.
- [License-Plate_VN_Multi, v2](https://universe.roboflow.com/lp-read-capstone-project/license-plate_vn_multi/dataset/2):
  đã kiểm tra ZIP 9.139 ảnh, MIT, không preprocessing/augmentation. Bản YOLOv11
  không khởi động tải được; bản YOLOv5 đã tải được, sử dụng cùng loại bounding
  box để hỗ trợ đề xuất góc. Đây không phải chuyển sang train model YOLOv5.

Giữ README attribution/giấy phép và provenance ZIP của từng nguồn. Hai nguồn có
thể trùng nhau. Nhãn YOLOv11 tải về là bounding box, **không phải bốn góc thật**;
contour trong box và model chỉ đề xuất để người duyệt. Không tải ảnh khách hàng
lên Roboflow. Không đếm crop, ảnh tăng cường, ảnh không thấy biển hoặc ảnh chưa
duyệt như 50.000 ảnh train hợp lệ.

## Kiểm tra crop trước OCR

Chạy `expansion_crop_ocr.py --workspace <workspace> --follow` thay cho worker
`ocr-all` cũ; không chạy hai worker OCR cùng lúc. Worker ưu tiên validation
OCR chưa chắc, sau đó các ảnh mới chưa chắc và phần hàng đợi còn lại.

Thứ tự vùng cắt: nhãn đã duyệt/nhãn v3 khóa → YOLO bốn điểm trong bounding box
nguồn → contour/box/crop ACS bổ sung. Không tự sửa bốn góc của ảnh gốc.
Kiểm tra kích thước thật, tứ giác, tương phản và hàng chữ bằng text detector
trước khi gọi bộ nhận dạng. Crop thiếu chữ có một lần thử tăng tương phản;
OCR chưa rõ được thử tương phản/làm nét. Đạt kiểm tra crop chưa có nghĩa là
biển số đúng: vẫn cần hai cách OCR cùng đọc rõ và không có kết quả mâu thuẫn.

Crop dùng OCR được lưu riêng trong `<workspace>/ocr-crops/` và hiển thị trên
trang duyệt kèm nguồn cắt. Crop không đạt không hiện lại như một crop hợp lệ;
ảnh gốc vẫn giữ để kiểm tra trong bộ lọc **OCR không chắc / chưa đọc**.
Giữ bằng chứng OCR cũ để đối chiếu rò rỉ thận trọng. Không dùng chữ OCR cũ để
vượt qua kiểm tra crop mới hoặc tự xác nhận nhãn/biển số của người duyệt.

Không xóa tay ảnh khi copy/OCR còn chạy. Nút **Loại** của trang mở rộng ghi trạng
thái loại, không đồng nghĩa xóa file gốc. Xóa nguồn cần đối chiếu đúng ID, cập
nhật trạng thái và ngừng công việc liên quan; không xóa thư mục ACS/imports
hoặc crop hàng loạt theo tên file hay kết quả model chưa kiểm tra.

## Tự xác nhận định danh validation (đã được chủ dữ liệu cho phép)

Chạy `expansion_auto_identity.py --workspace <workspace> --follow` song song với
worker kiểm tra crop. Worker ưu tiên kiểm tra nốt validation và tự xác nhận
biển số khi crop đạt, một vùng biển, hai phương pháp đọc khác nhau cùng trả về
một biển hợp lệ và đều đạt ít nhất 95%. Không đoán O/0, I/1 hoặc ký tự thiếu.
OCR còn mâu thuẫn, có biển khác gần giống trong validation hoặc ảnh gần trùng
khác định danh vẫn để người duyệt. Nhiều ảnh cùng biển chính xác là cùng nhóm
xe, không tự coi đó là lỗi. Định danh người đã xác nhận luôn được giữ nguyên.

Phần định danh tự động có `identity_method=auto-ocr-v1`, bằng chứng/digest và
lịch sử version trong SQLite riêng. UI phân biệt số người duyệt với OCR tự
xác nhận; ảnh đã xác nhận ra khỏi bộ lọc **Chưa duyệt**. Nếu cần sửa, người
duyệt tải lại để nhận version mới, kiểm tra rồi lưu như bình thường. Lượt tự
động sẽ thu hồi xác nhận của chính nó nếu bằng chứng không còn đạt điều kiện.
Worker kết thúc khi validation đã qua lượt kiểm tra crop, để lại các ca không
đủ chắc. Không thay nhãn góc, ảnh, thành viên validation hoặc tự duyệt ảnh train.

## Cấu hình OCR đã đo và áp dụng ngày 24/10/2026

Worker mở rộng dùng `expansion_crop_ocr.py --workspace <workspace> --follow
--ocr-threads 4 --opencv-threads 1` (viết trên cùng một dòng khi chạy).
Chỉ chạy một worker OCR. Nếu không truyền tham số, mặc định tương thích vẫn
là hai luồng OCR và cấu hình OpenCV hiện có.

Cấu hình bốn luồng đã được thử trên 48 ảnh sao chép thuộc tám nhóm: nhanh hơn
18–29% so với hai lượt mốc, kết quả OCR khớp với sai số số thực tối đa 0,0001
và hash crop trùng nhau. Đây là kết quả trên mẫu, không bảo đảm mức tăng tốc
cho mọi hàng đợi. Model, ngưỡng, kiểm tra crop và hai cách đọc giữ nguyên.
Khởi động lại tiếp tục từ ảnh chưa có kết quả hiện hành; không xóa OCR để chạy
lại toàn bộ, không ghi đè định danh/góc đã duyệt hoặc thay bộ validation.

## Kéo góc và ảnh nắn xem trước

Kéo một trong bốn điểm trên ảnh trái để ảnh nắn bên phải cập nhật ngay.
Ảnh nắn là xem trước từ ảnh đang hiển thị; thao tác này không chạy lại OCR.
Các chữ OCR và điểm tin cậy bên dưới vẫn là kết quả đã xử lý trước đó.

Với validation/train v3, có thể kéo hoặc vẽ khung để xem trước, nhưng góc này
không được lưu. Lưu/Duyệt chỉ cập nhật định danh theo quy trình hiện có; nhãn
góc gốc giữ nguyên. Chuyển ảnh hoặc tải lại sẽ bỏ góc xem trước.
Với ACS/Roboflow, góc sửa được lưu khi bấm Lưu/Duyệt hoặc chuyển ảnh theo
cơ chế lưu bản nháp hiện có. Vẫn cần kiểm tra và xác nhận trước khi duyệt.

## Sửa góc lệch đã đối chiếu ảnh gốc

Công cụ `apply_corner_corrections.py` nhận kế hoạch sửa từng ảnh đã kiểm tra
bằng mắt, ghi tọa độ vào bản nháp và giữ lịch sử trước khi sửa. Chỉ áp dụng
cho ảnh ACS/Roboflow một biển, chưa duyệt góc. Ảnh đã duyệt, đã loại và nhãn
validation/train v3 được bảo vệ; có thay đổi phiên bản hoặc hash thì dừng.
Kế hoạch, ảnh đối chiếu và audit phải nằm ngoài Git.

Ảnh đổi góc được đưa lên đầu hàng đợi OCR hiện có để đọc lại đúng vùng mới.
OCR cũ chỉ còn là bằng chứng đối chiếu trùng xe; không dùng như kết quả hiện
hành. Việc sửa góc không tự xác nhận biển số hoặc duyệt nhãn train. Tab đang
mở ảnh cũ cần tải lại để nhận tọa độ và phiên bản mới trước khi lưu tiếp.

## Hàng đợi kiểm tra góc sau khi quét toàn bộ

Chọn nguồn ACS hoặc Roboflow, rồi chọn **Góc cần bạn kiểm tra**. Bộ lọc này
giữ lại ảnh chưa đủ bằng chứng về mép biển, nhiều vùng phát hiện, thiếu khung
hoặc không đọc được ảnh. Những khung đã được đối chiếu rõ từng ảnh có thể
được tách khỏi hàng đợi bằng chỉ mục riêng `corner-review-index.json` trong
workspace dữ liệu ngoài Git. Chỉ mục khớp đúng phiên bản, hash ảnh và tọa độ;
ảnh đổi phiên bản/tọa độ sẽ trở lại hàng đợi. Thiếu hoặc hỏng chỉ mục cũng
không làm ảnh chưa chắc biến mất. Việc tách hàng đợi không thay thế xác nhận
biển số, đối chiếu validation, duyệt nhãn train hoặc xuất bộ dữ liệu.

## Duyệt góc riêng khi chưa đọc được chữ hoặc chưa kiểm tra trùng

Chọn **ACS tháng 1** hoặc **Roboflow Việt Nam**, rồi **Góc cần bạn kiểm tra**.
Nhập người duyệt, chỉnh bốn điểm khớp mép biển, tích **Ảnh có đúng một biển
Việt Nam, thấy đủ bốn góc**, rồi bấm **Duyệt góc**. Trong bộ lọc góc, Enter
cũng thực hiện thao tác này khi con trỏ không ở ô nhập chữ. Ảnh được lưu và
chuyển tiếp; khi tải lại, ảnh đã duyệt góc không còn trong bộ lọc góc.

Không cần đoán biển số bị mờ, tích xác nhận chữ hoặc kết luận trùng để duyệt
góc. Thao tác này chỉ lưu góc và người duyệt; biển số, kết luận trùng và OCR
đã lưu giữ nguyên. Các thay đổi chữ/kết luận chưa lưu trên màn hình không
được xác nhận bằng nút này. Ảnh nghi trùng vẫn nằm trong hàng đợi trùng;
ảnh chưa rõ chữ vẫn cần kiểm tra OCR. Nếu mép biển cũng mờ thì chuyển ảnh,
không duyệt góc. **Lưu** chỉ lưu bản chỉnh sửa, không phải duyệt góc.

**Duyệt đầy đủ** vẫn yêu cầu xác nhận biển số và xem ảnh gần trùng theo quy
trình hiện có. Chỉ duyệt góc chưa làm ảnh được duyệt đầy đủ hay đủ điều kiện
đưa vào train. Validation/train v3 tiếp tục khóa nhãn góc gốc.


## Cập nhật trang duyệt ngày 2026-10-28

Trang duyệt nhớ ID ảnh, số thứ tự và bộ lọc trên trình duyệt. Tải lại sẽ tìm ID trong bộ lọc hiện tại; nếu ảnh đã rời bộ lọc, quay về vị trí gần nhất còn hợp lệ. Ô Đi đến mở trực tiếp số thứ tự. Ảnh xem trước kế tiếp được tải trước; thống kê OCR của màn không lọc OCR tính trong trang hiện tại để tránh đối chiếu lại toàn bộ hàng đợi.

Chọn nhóm đích Xe máy/Ô tô/Chưa phân loại rồi Chuyển nhóm. Cần nhập người duyệt; thao tác chặn ảnh đã loại, ảnh trùng, train/validation khóa và phiên bản cũ. Nhãn góc, định danh và nguồn ACS/Roboflow giữ nguyên. File được sao chép và kiểm tra hash trước khi đổi đường dẫn; bản trước được giữ ở vùng quản lý để khôi phục. Không tự suy luận loại xe từ dạng biển.

### Sửa preview sau chỉnh góc

Preview bên phải luôn được dựng từ bốn góc hiện tại khi ảnh mở xong hoặc khi kéo góc, không dùng crop OCR cũ làm preview hình học. OCR lưu trước vẫn chỉ là thông tin tham khảo. Crop xem trước giới hạn 160 px chiều rộng để phù hợp khung hiển thị và giảm xử lý trên giao diện. Cache một hàng đợi được vô hiệu khi SQLite/WAL hoặc chỉ mục góc thay đổi; không dùng cache để bỏ qua kiểm tra phiên bản lúc lưu.
