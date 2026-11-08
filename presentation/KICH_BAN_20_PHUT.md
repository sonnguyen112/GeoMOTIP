# Kịch bản thuyết trình — 20 phút, gồm video

25 slide chính, 6 phụ lục. Lời trình bày: 19 phút 15 giây; video demo: 45 giây.

## Slide 1. MOTIP: từ phát hiện đến duy trì danh tính

Mốc 00:00 · khoảng 20 giây

Em trình bày bài báo Multiple Object Tracking as ID Prediction của Gao và cộng sự, công bố tại CVPR 2025. Đồ án dựa trên mã nguồn tác giả, tái hiện pipeline và đánh giá thêm một nhánh hình học nhỏ. Phần thực nghiệm là các run đã huấn luyện và được đối chiếu artifact, không phải số liệu lấy từ bảng SOTA. Em sẽ đi từ bài toán, đầu vào và đầu ra, cách MOTIP liên kết đối tượng, phần bổ sung hình học, rồi đến thiết kế và kết quả thực nghiệm.

## Slide 2. Phát hiện đúng người chưa đảm bảo đúng ID

Mốc 00:20 · khoảng 35 giây

Mỗi frame có thể phát hiện đúng tất cả người, nhưng khi hai người đi ngang qua nhau thì tracker vẫn có thể đổi danh tính. Vì vậy phát hiện và liên kết là hai vấn đề khác nhau. Ví dụ một ID ban đầu theo người A rồi chuyển sang người B là lỗi ID switch. Trên hình, nhãn ID 0 được giữ trên cùng một quỹ đạo ở ba thời điểm; đây là ID do tracker gán, không phải nhãn danh tính thật. Thử thách chính của DanceTrack là ngoại hình tương tự, chuyển động thay đổi và che khuất. Các ID ở đây là định danh tạm cho quỹ đạo, không phải nhận dạng cá nhân.

## Slide 3. Đầu vào là video; đầu ra là hộp và ID theo thời gian

Mốc 00:55 · khoảng 60 giây

Đầu vào là một chuỗi frame của video. Ở mỗi frame, detector tìm các đối tượng và tạo hộp cùng đặc trưng; MOTIP dùng lịch sử các track để gán ID cho từng phát hiện. Đầu ra là các hộp kèm ID ổn định qua thời gian, tức các quỹ đạo đã được liên kết. Hình tổng quan cho thấy vòng xử lý: phát hiện, dự đoán ID từ lịch sử, rồi ghi quan sát mới sau quyết định. Nhãn ID 0 trong ví dụ đầu ra là ID do tracker dự đoán, không phải nhãn GT. Một ID biểu thị cùng một đối tượng trong video, không phải danh tính cá nhân ngoài đời. Phần hình học được bổ sung ở các slide sau, tại bước chấm mức phù hợp giữa phát hiện và track ứng viên.

## Slide 4. DanceTrack tách độ khó liên kết khỏi ngoại hình

Mốc 01:55 · khoảng 40 giây

DanceTrack được thiết kế để làm liên kết khó hơn so với tình huống chỉ cần phân biệt áo quần. Workspace có 40 sequence train với 41.796 frame và 25 sequence validation với 25.508 frame. Đồ án dùng full validation cho các bảng đánh giá. Ba ngân sách train là các subset nguyên sequence, không lấy frame rời rạc. Do đây là validation và có một tập con dùng chọn checkpoint, em không coi kết quả là test độc lập hay đối chiếu trực tiếp với bảng test của paper.

## Slide 5. Detector tạo hộp, confidence và đặc trưng

Mốc 02:35 · khoảng 40 giây

Deformable DETR xử lý ảnh để sinh hộp, confidence và đặc trưng query đối tượng. Sau lọc ngưỡng, một phát hiện được biểu diễn bằng hộp chuẩn hóa cx, cy, w, h và vector 256 chiều. Đây không phải embedding ID cuối cùng. Trong thực nghiệm môn học, em dùng detector tiền huấn luyện được giữ cố định để tập trung đánh giá việc liên kết. Paper có thiết lập end-to-end; kết quả của em không được mô tả như đã tái lập hoàn toàn ngân sách huấn luyện đó.

## Slide 6. Lịch sử được tổ chức theo track và thời gian

Mốc 03:15 · khoảng 55 giây

Lịch sử có dạng batch, group, time, track, feature. Group ở đây là các nhóm tăng cường quỹ đạo trong huấn luyện, không phải số người. Mỗi track có quan sát ở nhiều thời điểm nhưng có thể bị thiếu; mask giúp attention bỏ qua những vị trí không hợp lệ. Module trajectory modeling gồm feed-forward, residual và LayerNorm để thích nghi đặc trưng detector. Không nên nhầm nó với một mô hình dự đoán vị trí tương lai. Khi suy luận, lịch sử này được xây từ ID do tracker tự dự đoán.

## Slide 7. ID embedding cho biết quan sát thuộc slot nào

Mốc 04:10 · khoảng 50 giây

Một token lịch sử ghép đặc trưng 256 chiều với ID embedding 256 chiều. Phát hiện mới thì chưa biết ID nên dùng embedding rỗng học được. Token vào decoder do đó có 512 chiều. ID slot được hoán vị trong huấn luyện để mô hình không ghi nhớ rằng một người luôn mang số cụ thể. Cấu hình này có 50 slot và thêm lớp NEW. ID slot được tái sử dụng nhưng ID toàn cục dùng xuất quỹ đạo là một khái niệm khác, nên phải quản lý reset lịch sử khi tái cấp slot.

## Slide 8. Attention liên kết phát hiện với lịch sử

Mốc 05:00 · khoảng 70 giây

Self-attention giúp các phát hiện hiện tại tương tác với nhau; cross-attention để chúng truy vấn các token lịch sử có ID. Thông tin thời gian tương đối được thêm vào attention, còn mask chặn các quan sát hiện tại hoặc tương lai. Có sáu lớp decoder trong cấu hình được dùng; mã nguồn bỏ self-attention ở lớp đầu nên hình ở đây là sơ đồ chức năng, không khẳng định mọi lớp giống hệt nhau. Đầu phân loại sinh logit cho 50 slot và NEW. Lưu ý đây vẫn là một bài toán dự đoán trong ngữ cảnh, không có một classifier riêng cho mỗi người của dataset.

## Slide 9. Huấn luyện ID theo ngữ cảnh, không nhớ ID toàn cục

Mốc 06:10 · khoảng 60 giây

Quy trình lấy clip có nhãn, ghép phát hiện với GT và tạo nhãn slot cục bộ. Các hoán vị và tăng cường mô phỏng quỹ đạo thiếu hoặc sai. Loss ID là cross-entropy trên vị trí hợp lệ; padding và frame chưa có ngữ cảnh không được tính như các query bình thường. Với MOTIP nền, detector đóng băng còn trajectory modeling, ID Decoder và ID embedding được học. Việc dùng GT ở bước tạo nhãn train là hợp lệ; khi inference không được dùng GT để nối track. Đây là ranh giới cần phân biệt rõ khi giải thích tiền huấn luyện geometry ở phần sau.

## Slide 10. Suy luận là một vòng đọc → quyết định → ghi

Mốc 07:10 · khoảng 60 giây

Ở frame đầu, các phát hiện đủ điều kiện được khởi tạo track mới. Sau đó tracker dự đoán ID từ lịch sử, giải quyết xung đột khi các phát hiện cùng thích một slot, và dùng NEW khi cần sinh track. Cấu hình đang đánh giá dùng object-max, không phải luôn dùng Hungarian ở bước suy luận. Các ngưỡng là detector 0,3, newborn 0,6 và ID 0,2. Track có thể thiếu quan sát trong cửa sổ 30 frame, nhưng khi slot được tái sử dụng thì lịch sử cũ phải bị xóa. Cuối cùng chỉ ảnh và trạng thái dự đoán đi vào bước suy luận, không dùng GT.

## Slide 11. Bổ sung geometry như một tín hiệu tương thích riêng

Mốc 08:10 · khoảng 35 giây

MOTIP có thể học quan hệ liên kết từ đặc trưng, nhưng với dữ liệu hạn chế, cung cấp quan hệ hình học trực tiếp là một giả thuyết đáng kiểm tra. Chúng em không thay ID Decoder hay thêm một Transformer lớn. Với mỗi phát hiện và mỗi track hiện hữu, nhánh nhỏ kiểm tra vị trí, kích thước và lịch sử gần đây có phù hợp hay không. Đây là giả thuyết và một cải tiến thực nghiệm của đồ án; không tuyên bố ý tưởng geometry trong tracking là hoàn toàn mới, cũng không tuyên bố nó luôn có lợi ở mọi quy mô dữ liệu.

## Slide 12. G4 mã hóa lịch sử hộp thành mô tả hình học 66 chiều

Mốc 08:45 · khoảng 65 giây

G4 tạo một mô tả cho từng cặp phát hiện hiện tại và track ứng viên. Vector 66 chiều gồm hộp hiện tại và bốn hộp lịch sử; độ dời tương đối và log thay đổi kích thước; vận tốc, phần dư ngoại suy và biến thiên vận tốc; log thay đổi diện tích; mã lag và các validity mask. Hộp tại t−4 phải thực sự cách bốn frame, không phải phần tử thứ tư trong danh sách quan sát. Nếu thiếu lag thì giữ mask bằng 0, không thay bằng quan sát gần nhất. Trong code, độ dời được chuẩn hóa theo kích thước hộp hiện tại. Các đặc trưng này được tính trong hệ tọa độ ảnh.

## Slide 13. G4–MLP bổ sung điểm hình học vào logit ID

Mốc 09:50 · khoảng 60 giây

Với mỗi track ứng viên, điểm hình học được kết hợp với logit mà MOTIP dự đoán cho cùng track. G4 có 66 chiều, đi qua MLP 66–128–128–1 với GELU và 25.217 tham số. Mean/std chỉ tính từ train. Fusion cố định là logit MOTIP chia hai cộng logit geometry chia 0,5. Với NEW, geometry đóng góp bằng không. MOTIP vẫn quyết định việc tạo track mới.

## Slide 14. Tiền huấn luyện: đổi nguồn hộp, giữ cùng một MLP

Mốc 10:50 · khoảng 60 giây

Bước đầu học bốn epoch GT–GT: tọa độ hộp hiện tại và lịch sử đều lấy từ GT train. Mẫu cùng người mang nhãn một, khác người mang nhãn không. Bước sau giữ MLP và học tiếp bốn epoch trên hộp detector đóng băng. Mỗi frame, hộp dự đoán được ghép với GT train bằng Hungarian với IoU ít nhất 0,5; GT chỉ cung cấp danh tính để nối lịch sử và tạo nhãn. Tọa độ đầu vào vẫn là tọa độ detector, không bị thay bằng GT. Các hộp không ghép được bị loại. ID Decoder không tham gia tạo cache; validation không tham gia fit hoặc chuẩn hóa. Khi suy luận thật, việc nối lịch sử dựa vào ID mô hình dự đoán.

## Slide 15. Hàm mất mát tổng giám sát cả hai đầu ra

Mốc 11:50 · khoảng 55 giây

Sau tiền huấn luyện, nhánh hình học được tích hợp vào tracker. Hàm mất mát tổng bằng L_fused cộng L_geom, hai trọng số đều bằng một. L_fused là CE trên logit ID đã hợp nhất; L_geom là CE trực tiếp trên các ứng viên track hiện hữu. Chỉ query có target track và lịch sử lag hợp lệ tham gia L_geom; NEW không nhận giám sát hình học. Giám sát trực tiếp giúp MLP nhận gradient mà không chỉ qua đường fusion.

## Slide 16. Ba ngân sách lồng nhau, một tập đánh giá chung

Mốc 12:45 · khoảng 35 giây

Ba ngân sách lần lượt là 2.090, 4.179 và 6.265 frame, tương đương khoảng năm, mười và mười lăm phần trăm của 41.796 frame train. Có ba, sáu và tám sequence, các tập nhỏ nằm trong tập lớn. Lựa chọn dựa trên metadata chứ không nhìn mức cải thiện. Tất cả được đánh giá trên cùng 25.508 frame validation. Chọn checkpoint theo một tập con cố định 3.894 frame, nhưng tập đó vẫn nằm trong full validation. Vì vậy em sẽ gọi đúng đây là kết quả validation, không phải test độc lập.

## Slide 17. Đối chiếu MOTIP nền với MOTIP + G4

Mốc 13:20 · khoảng 35 giây

Trong từng ngân sách, MOTIP nền và MOTIP + G4 dùng cùng detector, batch một, accumulation bốn, FP32, AdamW và lịch học. Ba seed là 42, 123, 3407; mỗi run mười epoch. Điểm khác chủ đích là nhánh geometry và bước tiền huấn luyện của nó. Các run được tái sử dụng có phép kiểm tra nguồn của cả mười tám bảng summary. Cùng epoch không có nghĩa cùng số update giữa các ngân sách; và geometry có chi phí pretraining thêm. Đây là các giới hạn cần nói rõ khi diễn giải hiệu quả dữ liệu.

## Slide 18. MOTIP + G4 cải thiện HOTA, AssA và IDF1

Mốc 13:55 · khoảng 55 giây

Hình cho thấy đường của MOTIP + G4 nằm trên MOTIP nền ở cả ba ngân sách. AssA là chỉ số chính vì câu hỏi tập trung vào association. Các khoảng sai số là độ lệch chuẩn mẫu của ba seed, không phải confidence interval. Kết quả ở năm phần trăm tăng khoảng 7,36 điểm AssA, ở mười phần trăm khoảng 8,82, và mười lăm phần trăm khoảng 7,53. Không có một xu hướng đơn điệu hoàn hảo giữa ba mức, nên em không dùng hình này để kết luận một quy luật scaling tổng quát.

## Slide 19. Mức tăng AssA đạt +7,36 đến +8,82 điểm

Mốc 14:50 · khoảng 40 giây

Ở 5% dữ liệu, AssA trung bình của MOTIP nền là 17,32 và MOTIP + G4 là 24,68. Ở 10%, hai giá trị là 26,69 và 35,51; ở 15% là 30,73 và 38,25. Đây là chênh lệch điểm trên thang 0–100, không phải phần trăm tương đối. Tất cả checkpoint đều chọn theo cùng tiêu chí AssA trên small validation cố định, rồi đánh giá trên full validation.

## Slide 20. Cả chín cặp seed/ngân sách đều cải thiện AssA

Mốc 15:30 · khoảng 40 giây

Mỗi cột là MOTIP + G4 trừ MOTIP nền theo cùng seed và cùng ngân sách. Tất cả chín cặp đều dương về AssA và giảm IDSW. Nhưng ba ngân sách không độc lập hoàn toàn vì dữ liệu lồng nhau, và ba seed vẫn là số lượng nhỏ. Vì vậy em dùng cách diễn đạt là nhất quán trong các run đã quan sát, không khẳng định đã chứng minh thống kê hay luôn đúng trên mọi dataset. Dữ liệu ghép cặp cũng được bàn giao để tránh việc chỉ giữ những seed thuận lợi.

## Slide 21. ID switch giảm, chi phí tính toán vẫn có

Mốc 16:10 · khoảng 40 giây

IDSW trung bình giảm ở cả ba mức dữ liệu, khoảng ba mươi ba đến bốn mươi bảy phần trăm. MLP chỉ có 25 nghìn tham số nhưng vẫn phải dựng và chấm các cặp ứng viên. Các cặp seed 42 trên cùng máy cho thấy FPS giảm vài phần trăm; không lấy FPS của máy trường trộn với máy local để kết luận. Trong demo mới chạy lại trên cùng sequence, em cũng ghi throughput thực tế riêng. Video phát lại ở tốc độ gốc không có nghĩa mô hình suy luận realtime ở tốc độ đó.

## Slide 22. G4 giữ ID khi các đối tượng giao cắt

Mốc 16:50 · khoảng 40 giây

Ở đây em theo cùng người GT 2 qua ba frame 53, 56 và 57. Khung xanh là bbox dự đoán; nét đứt trắng là GT của người đó. MOTIP nền đổi ID 0 thành 2 ở frame 57, còn MOTIP + G4 giữ ID 0. ID nội bộ không cần trùng số ID GT: tiêu chí là một người giữ cùng nhãn qua thời gian. Đã kiểm tra cả chín frame quanh sự kiện và phép ghép CLEAR/TrackEval, không chỉ ba ảnh trên slide. Đây là ví dụ tích cực được chọn để minh họa, không phải mẫu ngẫu nhiên hay ước lượng tỷ lệ lỗi.

## Slide 23. G4 duy trì ID khi chồng lấn

Mốc 17:30 · khoảng 40 giây

Tình huống thứ hai là người GT 8 trong dancetrack0034, qua frame 737, 740 và 741. Các đối tượng mặc giống nhau và có bbox chồng lấn. MOTIP nền chuyển từ ID 387 sang 442, trong khi MOTIP + G4 giữ ID 6 trên toàn bộ cửa sổ chín frame. Cả hai đều có hộp khớp cùng đối tượng GT, nên khác biệt thể hiện ở tính liên tục danh tính. Đây tiếp tục là ví dụ chọn có chủ đích. Sau slide này em chiếu video demo tổng quát 45 giây; video không phải hai đoạn vừa phân tích.

### Chiếu video demo (45 giây)

Mốc 18:10 · demo/DEMO_MOTIP_DanceTrack.mp4

Đây là demo quy trình suy luận tổng quát, không phải hai cửa sổ GT vừa phân tích.

## Slide 24. Bằng chứng tích cực nhưng kết luận có phạm vi

Mốc 18:55 · khoảng 40 giây

Đã quan sát được geometry cải thiện association trong cả ba ngân sách và ba seed, với detector cố định. Chưa chứng minh nó luôn hơn khi full-data end-to-end hoặc ngoài DanceTrack. Geometry đã có pretraining riêng nên không được coi toàn bộ gain là chỉ do kiến trúc. Validation dùng chọn checkpoint còn nằm trong tập full validation, nên cần một test độc lập cho kết luận mạnh hơn. Detector tiền huấn luyện cũng là nguồn dữ liệu bổ sung; ít dữ liệu ở đây là ít giám sát association, không phải toàn hệ thống chưa nhìn dữ liệu khác.

## Slide 25. Liên kết học được + hình học tường minh

Mốc 19:35 · khoảng 25 giây

Tóm lại, phần chính của đồ án là hiểu và tái hiện MOTIP: detector tạo đặc trưng, decoder dùng lịch sử mang ID để dự đoán ID hiện tại, rồi cập nhật bộ nhớ. Nhánh bổ sung cung cấp geometry đa độ trễ, chấm từng cặp ứng viên bằng một MLP nhỏ và hợp nhất logit. Trên các subset và seed đã khảo sát, nó tăng AssA và giảm ID switch. Các giới hạn về pretraining, chọn checkpoint và dữ liệu được trình bày công khai. Bước tiếp theo nên là test độc lập và đối chứng ngân sách compute trước khi mở rộng kết luận.

