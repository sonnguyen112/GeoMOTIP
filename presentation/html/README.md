# Slide HTML và bản PowerPoint

Mở `index.html` bằng Chromium/Chrome/Edge. Dùng ←/→, PageUp/PageDown hoặc
Space để chuyển slide; Home về đầu. Có thể nhấn F11 để trình chiếu toàn màn
hình. Deck tự co theo cửa sổ; không cần kết nối Internet.

Hai font DejaVu Sans được kèm trong `fonts/`, có giấy phép. HTML tự đọc font
local. Với PPTX, nên cài font nếu máy chưa có, hoặc dùng bản PDF đã nhúng font.

Nguồn layout: `scripts/build_html_presentation.py`. Xuất lại bằng:

```bash
.venv-docs/bin/python scripts/build_slides.py
```

Playwright/Chromium chỉ cần lúc dựng hoặc kiểm tra. Người xem file HTML/PPTX
không cần cài môi trường Python. `assets/` lưu các sơ đồ riêng được raster hóa
ở mật độ cao, không phải ảnh nguyên trang. Biểu đồ, chữ và bảng trong PPTX
được xuất native để tiếp tục chỉnh sửa. Công thức chỉnh bằng nguồn `.tex`.

`BROWSER_PREVIEW.pdf`: bản xem từ Chromium. Bản PDF chính cạnh PPTX được xuất
từ PowerPoint bằng Impress để phát hiện lỗi chuyển đổi. `renders/` là ảnh QA
nội bộ, không cần đưa vào gói nộp.
