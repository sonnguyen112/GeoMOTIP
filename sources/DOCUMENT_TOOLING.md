# Công cụ tài liệu và trạng thái kiểm thử

## Báo cáo LaTeX

- Template: [kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style),
  mẫu preprint một cột, MIT; không phải mẫu chính thức của arXiv hoặc CVPR.
- Bản nguồn upstream và giấy phép giữ tại `report/latex/template/`.
- XeLaTeX hỗ trợ trực tiếp tiếng Việt Unicode. BibTeX quản lý sáu nguồn có
  thật; hình/bảng có caption, label, font nhúng trong PDF.
- 13 phương trình có đánh số, dùng `equation`/`aligned` và ký hiệu toán thật.
  `scripts/latex_math_sections.py` lưu rõ các đoạn kỹ thuật để tránh chuyển đổi
  tự động sai công thức; script kiểm đủ các đoạn khi tạo `content.tex`.
- Bộ biên dịch nằm trong Docker `motip-course-latex:local`, tách khỏi môi
  trường huấn luyện. Không dùng shell escape, không gửi bản thảo lên cloud.
- Tái lập: `scripts/build_latex.py`; log tại `report/latex/build/`.

## Skill thuyết trình

- [PPTX — anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/pptx).
- Cài tại `/home/sonnguyen112/.codex/skills/pptx` qua skill-installer.
- Đã đọc và áp dụng kiểm tra OOXML, layout/overflow, notes và render PDF.
  Đã bỏ đường/stripe trang trí, giữ nội dung và hệ màu thống nhất sẵn có;
  sửa lời nói về tỷ lệ giảm IDSW thành khoảng 33–47%, khớp bảng dữ liệu.
- Công thức slide do XeLaTeX dựng, xuất ảnh alpha 450 DPI bằng
  `scripts/build_math_assets.py`; nguồn đi kèm. Văn bản/card/bảng vẫn là đối
  tượng PowerPoint chỉnh sửa được, công thức là hình typeset (sửa bằng `.tex`).
- Chữ card, flow và ô bảng căn giữa hai chiều theo yêu cầu; header/footer giữ
  bố cục riêng ngoài box.
- Skill mới được Codex tự nhận ở lượt/phiên kế tiếp. Không cần skill để người
  nhận mở file PPTX/PDF.

## Thiết kế lại bằng HTML/CSS

- Bản chính hiện tại được dựng bằng `build_html_presentation.py` và
  `export_html_pptx.py`; lệnh `build_slides.py` chuyển tiếp vào hai bước này.
- HTML/CSS tạo bố cục ảnh rộng, filmstrip, token, memory bank, timeline và
  biểu đồ thay cho dãy card lặp lại. Xem `SLIDE_DESIGN_REFERENCES.md` để biết
  các nguồn thiết kế đã tham khảo trực tuyến và nguyên tắc đã áp dụng.
- Chromium/Playwright đo kích thước; không chụp nguyên slide làm PPTX.
  Chữ dùng DejaVu Sans để bảo toàn tiếng Việt trong bản PDF; font được kèm
  cùng giấy phép. Biểu đồ có dữ liệu spreadsheet và error bar native.
  Sơ đồ phức tạp, ảnh và công thức được chèn như asset riêng ở mật độ cao.
- Báo cáo nhúng PDF vector của hình, có nguồn Matplotlib dựng lại từ số đo
  và các rollout cũ. Không có huấn luyện mới hoặc ảnh tạo sinh giả thực nghiệm.

## MCP PowerPoint cục bộ

- [Office-PowerPoint-MCP-Server](https://github.com/GongRzhe/Office-PowerPoint-MCP-Server), MIT.
- Package `office-powerpoint-mcp-server==2.0.7`, dependency `mcp==1.30.0`.
  Repo upstream đã archive; pin phiên bản để không tự đổi API. Phiên bản MCP
  Python 2.x không tương thích FastMCP của server này, nên môi trường riêng
  được pin ở nhánh 1.x; không chỉnh dependency của mô hình nghiên cứu.
- Venv: `/home/sonnguyen112/.local/share/motip-powerpoint-mcp`.
- Đã đăng ký Codex với tên `powerpoint` bằng `codex mcp add`, không thay các
  server có sẵn. Chạy stdio, không mở HTTP/SSE và không cần API key.
- Đã kiểm tra handshake, 37 tool, đọc 29 slide, cập nhật metadata và lưu PPTX
  qua MCP. Kết quả: `sources/POWERPOINT_MCP_CHECK.json`.
- Kiểm thử lại bằng Python của venv MCP:

```bash
/home/sonnguyen112/.local/share/motip-powerpoint-mcp/bin/python scripts/check_powerpoint_mcp.py
```

- Phiên Codex đang mở có thể cần khởi động lại để nạp namespace tool mới.
  Việc server đã được kiểm thử qua MCP client độc lập không đồng nghĩa namespace
  mới đã tự xuất hiện trong phiên hiện tại.

PowerPoint MCP dùng để thao tác và kiểm tra tài liệu local. Nó không phải mô
hình sinh nội dung khoa học, không kiểm chứng thay tác giả các số liệu trong
slide và không phải thành phần của pipeline tracking.
