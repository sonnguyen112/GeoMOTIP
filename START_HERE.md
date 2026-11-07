# Mở bộ đồ án

**Học viên:** Nguyễn Hồ Trường Sơn — 25C01020  
**Giảng viên:** TS. Trương Phước Hưng  
**Môn:** Thị giác máy tính

## Các tệp nên mở trước

1. `report/BAO_CAO_MOTIP.pdf`: báo cáo khoa học một cột, 15 trang, biên dịch XeLaTeX.
2. `presentation/THUYET_TRINH_MOTIP.pptx`: slide chỉnh sửa được; 25 slide chính + 6 phụ lục.
3. `presentation/KICH_BAN_20_PHUT.md`: lời trình bày theo slide, tổng thời gian gợi ý 20 phút.
4. `demo/DEMO_MOTIP_DanceTrack.mp4`: video H.264 1080p, 45 giây, vào ngay cảnh tracking.
5. `personal_presentation/index.html`: bản trình chiếu HTML độc lập, ưu tiên khi dùng máy cá nhân. Phím F để toàn màn hình, mũi tên để đổi trang, O xem tổng quan, Z phóng to hình và V xem video. Bản này đi kèm font/hình/video offline và không chuyển qua PPTX.

Bản PDF của slide để dự phòng khi máy trình chiếu không có đúng font:
`presentation/THUYET_TRINH_MOTIP.pdf`.

Bản HTML/CSS: `presentation/html/index.html`, mở offline bằng trình duyệt và
dùng phím ←/→ để trình chiếu. PPTX được chuyển từ layout HTML nhưng giữ chữ,
bảng và biểu đồ kết quả chỉnh sửa được. Không cần Internet để trình chiếu.

Trước buổi báo cáo, đọc `presentation/HOI_DAP_10_PHUT.md`. Không cần đọc hết
31 slide khi trình bày; 6 slide cuối dành cho hỏi đáp. Demo bắt đầu ngay ở
00:00, không có phần giới thiệu; có thể chiếu toàn bộ 45 giây khi trình bày.

## Kết quả nào được dùng?

Bảng chính dùng 18 kết quả checkpoint của ba seed 42/123/3407, hai mô hình MOTIP nền/MOTIP + G4
và ba subset 5%/10%/15% của DanceTrack. Checkpoint chọn bằng AssA trên small
validation cố định, sau đó đánh giá trên full validation 25.508 frame. Không gọi
đây là test độc lập; không trộn các bảng legacy 2.088/9.547 frame khác protocol.

Các run huấn luyện được tái sử dụng có đối chiếu artifact, không phải được huấn
luyện mới trong đêm tạo tài liệu. Hai scene demo được suy luận mới từ ảnh và
trạng thái dự đoán; không dùng GT để quyết định ID.

## Cách chạy lại

Xem `README.md`. Chạy model cần CUDA và môi trường MOTIP; xem PDF/slide/video
không cần GPU. Nguồn chính của PDF là `report/latex/main.tex`, `content.tex` và
`references.bib`; đồng thời giữ Markdown và DOCX để chỉnh sửa.

MOTIP gốc là công trình của Gao, Qi và Wang (CVPR 2025). Mã tác giả được ghi
nhận và giữ giấy phép; phần tái hiện, geometry bổ sung và tích hợp/thực nghiệm
được phân biệt trong báo cáo. Không trình bày mã nguồn tác giả như viết mới từ
đầu, không gọi bảng validation là kết quả vượt SOTA trên test.
