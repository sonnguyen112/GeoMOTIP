# Bản LaTeX một cột

Template: [arxiv-style](https://github.com/kourgeorge/arxiv-style), giấy phép MIT,
lưu nguyên bản trong `template/`. Đây là mẫu preprint của cộng đồng, không phải
mẫu chính thức của arXiv hay dấu xác nhận công bố. Bản đồ án điều chỉnh khổ A4,
phông Unicode tiếng Việt, tiêu đề học phần, cỡ chữ và chú thích.

`main.tex`: bố cục và kiểu chữ. `content.tex`: nội dung/bảng được sinh từ bản
Markdown đã kiểm chứng. `references.bib`: các nguồn thực được sử dụng.

Từ thư mục gốc đồ án:

```bash
.venv-docs/bin/python scripts/make_latex.py
docker build -f scripts/latex.Dockerfile -t motip-course-latex:local scripts
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" motip-course-latex:local
cp report/latex/build/main.pdf report/BAO_CAO_MOTIP.pdf
```

Hoặc dùng TeX Live/MiKTeX có XeLaTeX và `latexmk`:

```bash
cd report/latex
latexmk -xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Không cần upload lên Overleaf; toàn bộ quy trình chạy cục bộ. `content.tex` chứa
công thức LaTeX thực, còn Markdown giữ cách viết dễ đọc trên terminal.
