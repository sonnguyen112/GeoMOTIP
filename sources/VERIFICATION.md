# Kiểm tra trước bàn giao — 06/11/2026

## Nội dung khoa học

- Đối chiếu 18 dòng số liệu với `pedestrian_summary.txt` gốc: HOTA, DetA, AssA,
  IDF1, IDSW khớp trong dung sai làm tròn 0,00051.
- Tính lại trung bình và độ lệch chuẩn mẫu từ ba seed, `ddof=1`; khớp tệp tổng
  hợp có sẵn. Cả chín cặp có ΔAssA, ΔHOTA, ΔIDF1 dương và IDSW thấp hơn.
- Kiểm tra protocol geometry của cả chín nhóm seed/ngân sách: sequence train
  đúng manifest và `validation_used=false`. Metadata cache GT-linked xác nhận
  không gọi ID Decoder khi tạo liên kết tiền huấn luyện.
- Đối chiếu công thức loss với mã thực thi: V3 dùng ID CE trên logit fused cộng
  geometry auxiliary CE; không ghi nhầm thành có thêm một CE nền độc lập.
- Bảng chính chọn checkpoint theo small-validation AssA, không thay bằng bảng
  best full-validation cũ. Báo cáo nêu rõ small validation nằm trong full val.

## Code và demo

- Bộ test geometry/runtime được bàn giao: **13 passed**, một warning
  `torch.meshgrid` về API tương lai, không phải lỗi chạy.
- Bốn lượt suy luận mới trên hai sequence đều có frame/ID khớp artifact gốc;
  sai khác tọa độ hộp lớn nhất bằng **0,0**. Xem `QUALITY_CHECK.json`.
- Script inference không đọc annotation. Việc chọn cảnh đông chỉ sử dụng
  metadata dataset, không dùng mức cải thiện E0/V3.
- Video đã kiểm tra container/codec bằng FFprobe: H.264, yuv420p, 1920×1080,
  20 FPS phát lại, 45,00 giây. Không có audio, màn giới thiệu hoặc outro;
  chú thích tiếng Việt có sẵn. Cửa sổ frame 201–600 của dancetrack0004 và
  201–700 của dancetrack0034 được giữ cố định, không chọn theo sai/đúng của ID.
- Throughput được ghi riêng trong `demo/inference*/<model>/complete.json`.

## Trình bày và gói nộp

- PDF báo cáo **13 trang**, A4 một cột, biên dịch bằng XeLaTeX + BibTeX;
  template `kourgeorge/arxiv-style` (MIT), lưu kèm nguồn và giấy phép.
  Đã xem bản render/contact sheet, kiểm tra công thức và dấu tiếng Việt.
  Log biên dịch không còn citation/reference chưa giải quyết, glyph thiếu
  hoặc overfull hbox. Hai cảnh báo underfull về giãn dòng không làm mất nội dung.
- 13 công thức chính dùng môi trường `equation`/`aligned`, có đánh số;
  ký hiệu vector, ma trận, chỉ số, phân số và tập hợp dùng math mode thực.
  Công thức slide được dựng bằng XeLaTeX và xuất nền trong suốt 450 DPI;
  không dùng chuỗi Unicode giả phân số/chỉ số. Chữ trong card, flow và ô bảng
  được căn giữa ngang/dọc, sau đó kiểm tra qua bản PDF xuất từ Impress.
- PowerPoint **29 slide**: 23 chính, 6 phụ lục. Tổng mốc thời gian gợi ý trong
  notes/kịch bản là 1.200 giây. Đã xuất PDF 29 trang bằng LibreOffice Impress và
  xem contact sheet, kiểm tra chữ tiếng Việt và vị trí các hình/bảng.
- Bản thiết kế lại dùng HTML/CSS và đo layout bằng Chromium; không có overflow
  chữ trong HTML. PPTX giữ 168 đối tượng chữ, 6 biểu đồ và 4 bảng native;
  không raster hóa nguyên slide. Kiểm lại dữ liệu biểu đồ với các trung bình
  ba seed từ nguồn đã kiểm chứng. Các error bar là độ lệch chuẩn mẫu.
- Font DejaVu Sans được kèm giấy phép để hỗ trợ tiếng Việt đầy đủ; đã kiểm
  tra trích xuất chính xác tên giảng viên trong PDF, không chỉ xem ảnh render.
  Máy chưa có font nên cài font kèm theo trước khi mở PPTX hoặc dùng PDF.
- Đã đổi 6 hình trong báo cáo sang PDF vector (nhúng ảnh thật khi cần), kèm
  SVG/PNG. Luồng MOTIP và bộ nhớ token chỉ là sơ đồ cấu trúc; không giả làm
  attention map thực nghiệm. Hình định tính phóng to cùng ROI ở hai tracker;
  ID lấy từ các dự đoán đã chạy, không dùng nhãn GT để tô thành kết quả đúng.
- Đã xem frame video ở cảnh đông người; không dùng annotation làm hình dự đoán.
- DOCX là bản có thể chỉnh sửa từ cùng nội dung; PDF chính lấy từ nguồn LaTeX,
  không còn dùng renderer ReportLab. Không khẳng định DOCX có bố cục giống PDF.
- Skill PPTX của Anthropic đã cài; validator OOXML của skill báo
  `All validations PASSED!`. PowerPoint MCP đã initialize/list tools/open/info
  và cập nhật core properties/save thành công qua stdio; 37 tool, 29 slide.
  Không upload nội dung lên dịch vụ ngoài. Xem `POWERPOINT_MCP_CHECK.json`.
- ZIP được kiểm CRC bằng `ZipFile.testzip()`. Gói tài liệu không chứa `.env`,
  cookie VPN, virtualenv, toàn dataset, trọng số lớn hay chuẩn bị MOT17 đã hủy.
- Không tải lên máy chủ hoặc tự nộp bài thay người học.

Đây là phạm vi kiểm tra thực hiện cho bộ đồ án, không phải cam kết rằng mọi dữ
liệu tiền huấn luyện detector công bố sẵn đã được kiểm toán độc lập. Những giới
hạn về detector, validation selection và compute được giữ trong báo cáo.
