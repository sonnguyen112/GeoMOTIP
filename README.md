# MOTIP và hình học quỹ đạo đa độ trễ — đồ án Thị giác máy tính

**Nguyễn Hồ Trường Sơn — 25C01020**  
**Giảng viên: TS. Trương Phước Hưng**  
Hạn nộp: 11:00 ngày 07/11/2026. Báo cáo và slide bằng tiếng Việt.

Mở `START_HERE.md` để dùng ngay báo cáo, slide và video.

## Clone mã nguồn

```bash
git clone https://github.com/sonnguyen112/GeoMOTIP.git
cd GeoMOTIP
```

Mã MOTIP/TrackEval được giữ attribution theo giấy phép gốc. Dataset DanceTrack và
checkpoint không nằm trong repository; cần tải/được cấp quyền riêng. Gói `12.zip`
đính kèm các tài liệu đã xuất và video demo để xem độc lập với mã nguồn.

## Phạm vi và thư mục

Đồ án tái hiện MOTIP dựa trên mã nguồn Gao, Qi, Wang (CVPR 2025), kèm nhánh G4
theo cặp detection–track. Các kết quả DanceTrack 5%/10%/15%, ba seed được tái sử
dụng từ run đã hoàn tất, đối chiếu summary gốc. Không chạy huấn luyện mới trong
đợt chuẩn bị này. Kế hoạch MOT17 đã hủy; tệp dở giữ trong `_cancelled_mot17/`,
không đưa vào bài nộp.

```text
report/          PDF XeLaTeX một cột, nguồn .tex/.bib, DOCX và Markdown
presentation/    PPTX, PDF, nguồn HTML/CSS, kịch bản 20 phút và hỏi đáp
demo/            Video, dự đoán mới chạy và metadata tốc độ
figures/         Hình thuật toán và biểu đồ từ số đo
src/motip/       Snapshot MOTIP/TrackEval và phần tích hợp
configs/         Cấu hình inference hai checkpoint demo
experiments/     Protocol, manifest, số đo từng seed
sources/         Attribution, protocol, summary/log gốc
tests/           Bộ kiểm thử geometry/runtime có ghi nhận
scripts/         Chạy demo, kiểm thử, tạo tài liệu/video
checkpoints/     Trọng số local; tách khỏi gói tài liệu nhẹ
data/DanceTrack/ Symlink dữ liệu local, không sao chép toàn dataset
```

## Chạy lại demo

Môi trường tham chiếu: Linux/WSL, Python 3.12, PyTorch 2.4.0 + CUDA 12.1,
RTX 4060 Ti 16 GiB. Tạo môi trường MOTIP, cài dependency từ
`requirements-inference.txt`, đặt dataset tại `data/DanceTrack` và checkpoint
được cấp tại `checkpoints/`, sau đó chạy:

```bash
cd /home/sonnguyen112/data/Trail-MOT/MOTIP_CV_Course_Project
conda activate MOTIP
python scripts/run_tests.py
python scripts/run_demo_inference.py --model both
python scripts/run_demo_inference.py --model both --sequence dancetrack0034 --output demo/inference_crowded
.venv-docs/bin/python scripts/render_demo.py
```

Inference không đọc annotation. Nếu có `complete.json`, giữ kết quả hoàn tất
thay vì chạy chồng; muốn rerun thật thì chuyển thư mục kết quả sang tên backup.
Hai checkpoint train 15%, seed 42 là E0 epoch 6 / V3 epoch 9, chọn bằng AssA
trên small validation. Script tự rebase đường dẫn theo thư mục dự án khi di
chuyển. Máy khác cần dữ liệu tại `data/DanceTrack`, hai checkpoint và file
`checkpoints/g4_train_only_stats.pt` có checksum/provenance đúng.

Cài dependency theo `requirements-inference.txt` trong môi trường riêng; wheel
CUDA/ABI cần phù hợp hệ thống. Extension deformable attention kèm theo chỉ dùng
cho Python 3.12 Linux hiện tại. Khác môi trường thì biên dịch lại:

```bash
cd src/motip/models/ops
python setup.py build_ext --inplace
```

Không dùng `.so` Linux trong Python Windows native. Checkpoint tracker đã chứa
trọng số detector, không cần tải detector lại chỉ để inference.

## Mã huấn luyện và protocol

`src/motip/train.py`, `data/`, `models/motip/` và `submit_and_evaluate.py` là
pipeline được bàn giao, giữ attribution. Protocol, manifest và log thật nằm
trong `sources/artifacts/` và `experiments/`. Đường dẫn gốc trong các JSON là
hồ sơ provenance, không phải lệnh tự động khởi chạy.

Không dùng config inference để tuyên bố tái lập nguyên run training. Muốn train
lại chính xác phải khôi phục common initialization, geometry pretrain và
optimizer/schedule theo protocol. Đợt đồ án này không tự huấn luyện lại.

Checkpoint chính chọn trên 3.894 frame small validation rồi đo full validation.
Small validation nằm trong full validation nên không gọi đây là test độc lập.
Không trộn test ẩn, bảng SOTA paper hay bảng legacy khác protocol vào kết quả.

## Tạo lại tài liệu

Môi trường tài liệu tách khỏi GPU:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install -r requirements-documents.txt
.venv-docs/bin/python -m playwright install chromium
docker build -f scripts/latex.Dockerfile -t motip-course-latex:local scripts
.venv-docs/bin/python scripts/build_math_assets.py
.venv-docs/bin/python scripts/make_assets.py
.venv-docs/bin/python scripts/build_report.py
.venv-docs/bin/python scripts/build_latex.py
.venv-docs/bin/python scripts/build_slides.py
```

Báo cáo PDF được biên dịch thật bằng XeLaTeX + BibTeX, dùng template một cột
`kourgeorge/arxiv-style` (MIT), điều chỉnh A4 và Unicode tiếng Việt. Nguồn
chỉnh sửa nằm tại `report/latex/`; xem README ở đó để biên dịch không cần Docker
nếu đã có TeX Live/MiKTeX. `build_report.py` chỉ tạo DOCX, không ghi đè PDF LaTeX.
Nguồn Markdown/DOCX vẫn được giữ để đọc/sửa tiện lợi. Slide được thiết kế bằng
HTML/CSS tại `presentation/html/index.html`, xem bằng trình duyệt, phím ←/→
để chuyển trang. `build_slides.py` gọi quy trình HTML → đo layout bằng Chromium
→ xuất PPTX. Chữ Aptos, bảng và 6 biểu đồ kết quả vẫn là đối tượng PowerPoint
chỉnh sửa được; không chụp toàn slide thành ảnh. Hình/sơ đồ phức tạp dùng asset
riêng. PDF slide chính được xuất lại từ PPTX bằng LibreOffice Impress để QA.
HTML dùng hệ chữ sans-serif phổ biến; khi PowerPoint không có Aptos, có thể dùng
PDF chính để giữ bố cục đã kiểm tra.
Các phương trình trong slide dùng hình nền trong suốt từ XeLaTeX, 450 DPI,
có nguồn `.tex` tại `report/latex/math_assets/`; không phải công thức Unicode.
Chữ trong các box và ô bảng được căn giữa ngang/dọc.
Các hình trong báo cáo có bản PDF vector và SVG tại `figures/`; LaTeX nhúng
PDF, còn slide dùng PNG chất lượng cao. `FIGURE_PROVENANCE.json` phân biệt
sơ đồ minh họa với hình thực và bảng số đo. Mã dựng hình: `scripts/paper_figures.py`.
Video H.264/yuv420p 1080p, 45 giây, phát lại 20 FPS, không âm thanh, có chú thích Việt;
không có đoạn giới thiệu, outro hoặc dừng hình.
Đó là video trực quan từ suy luận thật, không phải quay màn hình hoặc nhãn GT.

`scripts/quality_check.py` kiểm toán provenance trong workspace gốc, tạo contact
sheet và kiểm tra tệp. Cần artifact nghiên cứu gốc để chạy phép kiểm toán đó;
không cần cho người chỉ xem bài nộp hoặc chạy inference bằng checkpoint.

Skill `pptx` từ `anthropics/skills` và PowerPoint MCP cục bộ đã được cài, kiểm
thử qua stdio; xem `sources/DOCUMENT_TOOLING.md`. Không cần tài khoản dịch vụ
ngoài để mở/tạo tài liệu. Các skill/MCP là công cụ soạn tài liệu, không phải
thành phần của thuật toán MOTIP.

## Nguồn và giấy phép

- MOTIP: https://github.com/MCG-NJU/MOTIP — Apache-2.0; giữ `src/motip/LICENSE`.
- TrackEval: https://github.com/JonathonLuiten/TrackEval — giữ attribution/license gốc.
- DanceTrack: https://github.com/DanceTrack/DanceTrack — không kèm toàn dataset trong bài nộp.

Không nhận detector, ID Decoder gốc, TrackEval hay dữ liệu là tự viết/tự thu
thập. Xem `sources/CODE_ATTRIBUTION.md` để phân định phần kế thừa và tích hợp.
