"""Explicit, audited LaTeX versions of the manuscript's mathematical passages.

Keys match source paragraph openings. No research numbers or model definitions
are inferred by the Markdown converter.
"""

PARAGRAPHS = {
    'Tại thời điểm t, đầu vào là ảnh': r"""Tại thời điểm $t$, đầu vào là ảnh $I_t$ và trạng thái lịch sử của tracker.
Detector tạo $N_t$ phát hiện. Với phát hiện thứ $n$, ký hiệu hộp, độ tin cậy và
đặc trưng lần lượt là
\begin{equation}
 \mathbf b_{t,n}=\begin{bmatrix}c_{x,t,n}&c_{y,t,n}&w_{t,n}&h_{t,n}\end{bmatrix}^{\mathsf T}
 \in\mathbb R^4,\qquad q_{t,n}\in[0,1],\qquad \mathbf f_{t,n}\in\mathbb R^{256}.
 \label{eq:detection}
\end{equation}
Hộp được chuẩn hóa theo kích thước ảnh. Đầu ra cuối cùng là tập hộp và ID toàn
cục dùng để nối các quan sát thành quỹ đạo.""",
    'Một tensor lịch sử điển hình có dạng': r"""Các tensor lịch sử được tổ chức theo batch, nhóm tăng cường, thời gian và track:
\begin{equation}
 \begin{aligned}
  \mathbf F_{\mathrm{hist}}&\in\mathbb R^{B\times G\times T\times K\times256},\\
  \mathbf B_{\mathrm{hist}}&\in\mathbb R^{B\times G\times T\times K\times4},\\
  \mathbf Y_{\mathrm{hist}},\ \mathbf M_{\mathrm{hist}}
   &\text{ có kích thước }B\times G\times T\times K.
 \end{aligned}
 \label{eq:history-tensors}
\end{equation}
Trong đó $B$ là batch, $G$ là nhóm tăng cường quỹ đạo, $T$ là số bước thời gian
và $K$ là số track tối đa sau padding. Số track thực tế có thể ít hơn $K$;
các phần tử padding không được tham gia attention hay loss. Không diễn giải
các chiều này thành số video hay số người cố định trên toàn tập dữ liệu.""",
    'Ở mức một attention head, có thể viết': r"""Ở mức một attention head, phép truy vấn ngữ cảnh được viết dưới dạng
\begin{equation}
 \begin{aligned}
  \mathbf A&=\operatorname{softmax}\!\left(
  \frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_h}}
   +\mathbf M+\mathbf B_{\mathrm{time}}\right),\\
  \operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)&=\mathbf A\mathbf V.
 \end{aligned}
 \label{eq:attention}
\end{equation}
$\mathbf Q$ được chiếu từ các token đang cần dự đoán; $\mathbf K$ và
$\mathbf V$ được chiếu từ ngữ cảnh; $d_h$ là số chiều của head.
$\mathbf M$ chặn phần tử padding và các quan hệ thời gian không hợp lệ,
còn $\mathbf B_{\mathrm{time}}$ là thiên lệch thời gian tương đối học được.
Trong self-attention, query, key và value cùng lấy từ tập phát hiện hiện tại;
trong cross-attention, key và value lấy từ lịch sử. Các head được kết hợp rồi
đi qua biến đổi feed-forward với residual và normalization. Cấu hình dùng sáu
lớp decoder; mã nguồn bỏ self-attention ở lớp đầu, nên sơ đồ attention được
hiểu ở mức chức năng chứ không giả định mọi lớp hoàn toàn giống nhau.""",
    'Có thể viết khái quát:': r"""Có thể viết khái quát đường dự đoán ID như sau:
\begin{equation}
 \mathbf z_t=\operatorname{Decoder}(\mathbf U_t,\mathcal H_{<t},\mathbf M),
 \qquad \boldsymbol\ell_B=\operatorname{IDHead}(\mathbf z_t).
 \label{eq:id-decoder}
\end{equation}
$\mathbf U_t$ là tập query hiện tại và $\mathcal H_{<t}$ là lịch sử trước thời
điểm $t$. Xác suất ID được suy ra từ softmax trên logit. Cách viết này mô tả
chức năng của pipeline; số lớp, kích thước và mask được lấy từ cấu hình/mã
nguồn đi kèm. Đây không phải bộ phân loại danh tính người trên toàn dataset.""",
    'Với mô hình nền, loss liên kết là': r"""Với mô hình nền, loss liên kết là cross-entropy trên tập query hợp lệ $\Omega$:
\begin{equation}
 \mathcal L_{\mathrm{ID}}=
 -\frac{1}{|\Omega|}\sum_{u\in\Omega}
 \log\left[\operatorname{softmax}(\boldsymbol\ell_{B,u})\right]_{y_u}.
 \label{eq:baseline-loss}
\end{equation}
$y_u$ là ID slot mục tiêu của query $u$. Frame đầu chủ yếu thiết lập ngữ cảnh;
những vị trí bị mask không được tính vào loss. Các tham số trajectory modeling,
ID Decoder và ID embedding được cập nhật, còn detector giữ nguyên. Đây là
thiết lập huấn luyện nhánh liên kết được chọn cho đồ án, khác với việc huấn luyện
toàn bộ hệ thống end-to-end trong mọi thí nghiệm của paper gốc.""",
    'Với phát hiện n và track ứng viên k,': r"""Với phát hiện $n$ và track ứng viên $k$, chúng tôi truy xuất hộp tại đúng các
thời điểm $t-\Delta$, với
\begin{equation}
 \mathcal D=\{1,2,4,8\},\qquad
 m_{\Delta}^{k}=\begin{cases}
  1,&\text{có quan sát của track }k\text{ tại đúng }t-\Delta,\\
  0,&\text{ngược lại.}
 \end{cases}
 \label{eq:exact-lags}
\end{equation}
Một vị trí thứ tư trong danh sách quan sát không nhất thiết là hộp ở $t-4$:
khi bị che khuất, thời điểm quan sát và chỉ số phần tử không trùng nhau.
Nếu một lag bị thiếu, khối đặc trưng tương ứng được điền giá trị trung tính
và đánh dấu validity bằng $0$; không thay bằng quan sát gần nhất.""",
    'Độ dời trong triển khai được chuẩn hóa': r"""Để viết gọn, trong các công thức sau, hộp hiện tại của phát hiện $n$ có tâm
$(c_{x,t},c_{y,t})$ và kích thước $(w_t,h_t)$; hộp lịch sử thuộc ứng viên $k$
được ký hiệu bằng chỉ số $t-\Delta$. Độ dời và thay đổi tỷ lệ là
\begin{equation}
 \begin{aligned}
 d_x^{\Delta}&=\frac{c_{x,t}-c_{x,t-\Delta}}{\max(w_t,\varepsilon)},&
 d_y^{\Delta}&=\frac{c_{y,t}-c_{y,t-\Delta}}{\max(h_t,\varepsilon)},\\
 d_w^{\Delta}&=\log\frac{w_t+\varepsilon}{w_{t-\Delta}+\varepsilon},&
 d_h^{\Delta}&=\log\frac{h_t+\varepsilon}{h_{t-\Delta}+\varepsilon}.
 \end{aligned}
 \label{eq:relative-geometry}
\end{equation}
Trong triển khai, $\varepsilon=10^{-5}$ và độ dời được chuẩn hóa theo kích thước
\emph{hộp hiện tại}. Các khối ứng với lag
không hợp lệ được đặt bằng $0$ và đi kèm validity mask riêng.""",
    'Các vận tốc giữa cặp lag liền kề': r"""Đặt $\mathbf p_t=(c_{x,t},c_{y,t})^{\mathsf T}$ và
$\mathbf s_t=(\max(w_t,\varepsilon),\max(h_t,\varepsilon))^{\mathsf T}$.
Với $\oslash$ là phép chia từng thành phần, vận tốc chuẩn hóa và phần dư ngoại suy là
\begin{equation}
 \begin{aligned}
 \mathbf v^{(a,b)}&=\frac{\mathbf p_{t-a}-\mathbf p_{t-b}}{b-a}\oslash\mathbf s_t,
 & (a,b)&\in\{(1,2),(2,4),(4,8)\},\\
 \mathbf r_v&=(\mathbf p_t-\mathbf p_{t-1})\oslash\mathbf s_t-\mathbf v^{(1,2)}.
 \end{aligned}
 \label{eq:velocity}
\end{equation}
Hai vector biến thiên vận tốc là $\mathbf v^{(1,2)}-\mathbf v^{(2,4)}$ và
$\mathbf v^{(2,4)}-\mathbf v^{(4,8)}$. Các đặc trưng chuyển động được xây dựng
trong hệ tọa độ ảnh và chỉ hợp lệ khi đủ các quan sát tương ứng.""",
    'Vector G4 được chuẩn hóa bằng mean/std': r"""Với vector G4 $\mathbf g_{nk}\in\mathbb R^{66}$, mean/std chỉ được tính từ
dữ liệu train. Sau chuẩn hóa, MLP dự đoán một điểm cho từng cặp phát hiện--track:
\begin{equation}
 \begin{aligned}
 \widetilde{\mathbf g}_{nk}&=\operatorname{Normalize}_{\mathrm{train}}(\mathbf g_{nk}),\\
 \ell_{G,nk}&=\mathbf W_3\,\phi\!\left(
       \mathbf W_2\,\phi(\mathbf W_1\widetilde{\mathbf g}_{nk}+\mathbf b_1)
       +\mathbf b_2\right)+b_3,\\
 \mathbf W_1&\in\mathbb R^{128\times66},\qquad
 \mathbf W_2\in\mathbb R^{128\times128},\qquad
 \mathbf W_3\in\mathbb R^{1\times128}.
 \end{aligned}
 \label{eq:geometry-mlp}
\end{equation}
Ở đây $\phi=\operatorname{GELU}$. Mạng có 25.217 tham số và không tạo một bộ
nhãn danh tính riêng trên toàn dataset. Chi phí tính toán tăng theo số cặp
phát hiện--track được chấm điểm.""",
    'Với mỗi track ứng viên, điểm hình học': r"""Với mỗi track ứng viên, điểm hình học được kết hợp với logit mà MOTIP dự đoán
cho cùng track. Công thức hợp nhất là
\begin{equation}
 \ell_{\mathrm{final},nk}=\frac{\ell_{B,nk}}{T_b}
                      +\lambda\frac{\ell_{G,nk}}{T_g},
 \qquad T_b=2,\quad T_g=0{,}5,\quad\lambda=1.
 \label{eq:fusion}
\end{equation}
Hình học chỉ chấm track hiện hữu có lịch sử hợp lệ; đóng góp của nó cho lớp
mới bằng không:
\begin{equation}
 \ell_{G,n,\mathrm{NEW}}=0,\qquad
 \ell_{\mathrm{final},n,\mathrm{NEW}}=\frac{\ell_{B,n,\mathrm{NEW}}}{T_b}.
 \label{eq:new}
\end{equation}
Các giá trị được giữ cố định, không hiệu chỉnh riêng theo seed hoặc ngân sách
sau khi xem kết quả validation. Khi tắt geometry, đường MOTIP nền được giữ
nguyên; phép chia nhiệt độ của MOTIP + G4 không phải một phần mặc định của MOTIP nền.""",
    'Trong phiên bản triển khai được đánh giá,': r"""Hàm mất mát tổng gồm hai thành phần:
\begin{equation}
 \begin{aligned}
 \mathcal L_{\mathrm{fused}}&=
    \operatorname{CE}_{\Omega}(\boldsymbol\ell_{\mathrm{final}},\mathbf y),\\
 \mathcal L_{\mathrm{geom}}&=
    \operatorname{CE}_{\Omega_G}(\boldsymbol\ell_G,\mathbf y),\\
 \mathcal L_{\mathrm{total}}&=\nu\mathcal L_{\mathrm{fused}}
                    +\mu\mathcal L_{\mathrm{geom}},\qquad\mu=\nu=1.
 \end{aligned}
 \label{eq:v3-loss}
\end{equation}
$\operatorname{CE}_{\Omega}$ là cross-entropy trung bình trên query hợp lệ;
$\Omega_G\subseteq\Omega$ chỉ giữ query có target là track hiện hữu và có
lịch sử lag hợp lệ. ID-loss weight bằng $1$. Loss detector không tạo cập nhật
do detector đóng băng. Loss ID được tính trên logit đã hợp nhất. Giám sát geometry trực tiếp
giúp nhánh mới nhận gradient mà không chỉ phụ thuộc đường residual đã nhân
hệ số hợp nhất.""",
    'HOTA được tính qua nhiều ngưỡng IoU.': r"""HOTA được tính qua nhiều ngưỡng IoU. Ở mỗi ngưỡng $\alpha$,
\begin{equation}
 \mathrm{HOTA}_{\alpha}
   =\sqrt{\mathrm{DetA}_{\alpha}\,\mathrm{AssA}_{\alpha}}.
 \label{eq:hota}
\end{equation}
Giá trị cuối là tổng hợp theo các ngưỡng, không lấy căn của hai giá trị DetA
và AssA đã trung bình rồi gọi đó là HOTA chính thức. AssA là chỉ số chính để
phân tích liên kết. IDF1 đánh giá tính nhất quán ID ở mức quỹ đạo; IDSW đếm các
sự kiện đổi ID theo CLEAR matching. HOTA, DetA, AssA và IDF1 càng cao càng tốt;
IDSW càng thấp càng tốt.""",
}
