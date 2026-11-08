# Chuẩn bị hỏi đáp — chọn trả lời ngắn trước, mở phụ lục khi cần

## 1. MOTIP khác một pipeline detector + ReID thông thường ở đâu?

MOTIP trực tiếp dự đoán ID cục bộ từ các phát hiện hiện tại và lịch sử mang ID. Nó không chỉ học một embedding rồi dùng khoảng cách cosine như một bước liên kết tách rời. ID Decoder học khai thác ngữ cảnh lịch sử để tạo phân phối ID.

## 2. ID slot có phải ID thật của người trong dataset không?

Không. Slot là ký hiệu cục bộ của tracker, được hoán vị trong huấn luyện và có thể tái cấp phát. ID toàn cục xuất ra để nối quỹ đạo là một ánh xạ khác. Slot được tái sử dụng thì bộ nhớ cũ phải được reset.

## 3. Có 50 slot tức tracker chỉ theo dõi được 50 người trong cả video?

Không phải tổng số người trong cả video. Đây là giới hạn từ vựng/bộ nhớ hoạt động trong cấu hình; slot được tái sử dụng khi track cũ kết thúc. ID toàn cục có thể lớn hơn 50. Nếu quá đông người hoạt động đồng thời, chính sách quản lý slot sẽ là giới hạn cần xét.

## 4. NEW hoạt động như thế nào?

NEW là lớp cho một phát hiện chưa phù hợp với ID hoạt động. Sau dự đoán và lọc newborn theo confidence, tracker cấp slot và ID toàn cục mới rồi thiết lập lịch sử. Geometry không được huấn luyện để phát minh một điểm NEW riêng trong cấu hình này.

## 5. Transformer của MOTIP nhìn gì?

Query là đặc trưng phát hiện hiện tại kèm embedding ID rỗng. Key/value lịch sử chứa đặc trưng các quan sát cũ ghép với embedding ID đã biết. Self-attention mô hình hóa quan hệ giữa phát hiện, cross-attention truy vấn lịch sử; mask và thông tin thời gian ngăn dùng tương lai. Cấu hình có sáu lớp decoder; trong mã nguồn, lớp đầu không dùng self-attention như các lớp sau.

## 6. Trajectory modeling có dự đoán chuyển động tương lai không?

Không phải một motion predictor riêng. Nó dùng FFN, residual và LayerNorm để thích nghi đặc trưng detector của lịch sử cho liên kết. Phần vận tốc/ngoại suy trong đồ án nằm ở mô tả geometry bổ sung.

## 7. Hungarian được dùng ở đâu?

Hungarian có vai trò ghép detection và GT khi huấn luyện detector. Trong cấu hình inference của nhóm thực nghiệm này, tracker dùng object-max kèm xử lý xung đột và ngưỡng. Không nói rằng mọi bước của MOTIP đều dùng hoặc đều không dùng Hungarian.

## 8. Vì sao cần nhánh geometry nếu Transformer có thể tự học?

Đó chính là giả thuyết cần đo, không phải điều mặc định đúng. Geometry đưa trực tiếp quan hệ vị trí/tỷ lệ nhiều lag vào một nhánh nhỏ. Trên các subset được thử nó có lợi, nhưng chưa chứng minh luôn có lợi khi Transformer được huấn luyện đầy đủ hơn.

## 9. G4 có những gì và vì sao đúng 66 chiều?

Mở slide phụ lục về G4: hộp hiện tại (4 chiều) + bốn hộp lịch sử (16) + validity lag (4) + vận tốc/ngoại suy (8) + validity vận tốc/ngoại suy (4) + biến thiên vận tốc (4) + validity tương ứng (2) + độ dời tương đối/log-scale (16) + log-area (4) + mã lag (4) = 66 chiều.

## 10. Mẫu số chuẩn hóa displacement là hộp nào?

Trong triển khai được đánh giá là kích thước hộp hiện tại: dx = (cx_t − cx_history) / max(w_t, ε). Cần giữ đúng công thức này, không đổi sang w_history chỉ vì một cách định nghĩa khác cũng hợp lý. Mean/std của vector lấy từ train.

## 11. Track bị che khuất thì lag t−4 lấy hộp nào?

Chỉ lấy hộp có timestamp đúng bằng t−4. Không có thì zero-fill và validity = 0; không lấy quan sát gần nhất hay phần tử thứ tư trong mảng. Nếu không, mô hình có thể nhầm mất quan sát thành một chuỗi chuyển động đều.

## 12. Dùng GT để nối bbox tiền huấn luyện có bị leak không?

Dùng nhãn train để tạo mục tiêu là hợp lệ. Chúng tôi kiểm tra manifest, protocol và cache metadata để giữ geometry pretraining trong các sequence train của từng ngân sách. Không dùng nhãn validation để fit MLP hoặc chuẩn hóa. Tuy nhiên lịch sử train GT-linked sạch hơn predicted-ID history khi inference; đó là distribution shift cần nêu, không được che đi.

## 13. Có thể nói hoàn toàn không data leak ở mọi thành phần không?

Không nên nói quá phạm vi kiểm tra. Đã kiểm tra phạm vi train của geometry và sự tách sequence association train/validation. Detector được nạp từ checkpoint công bố sẵn; chúng tôi không có toàn bộ log tiền huấn luyện detector để đưa ra một cam kết rộng hơn. Full validation cũng chứa tập chọn checkpoint nên không phải test độc lập.

## 14. Vì sao BCE lúc tiền huấn luyện nhưng CE khi tích hợp?

Tiền huấn luyện tạo các cặp phát hiện–track, gán nhãn cùng ID/khác ID nên dùng BCEWithLogits có trọng số dương. Khi tích hợp, mục tiêu là chọn đúng slot giữa các ứng viên nên dùng CE phụ. Cách lấy mẫu và phân phối nhãn của hai giai đoạn khác nhau.

## 15. Loss thực chạy có ba thành phần L_base + L_geom + L_fused không?

Không. Trong phiên bản này, ID criterion nhận logit đã hợp nhất, vì vậy loss thực là CE(l_final,y) + CE(l_G,y) với hai trọng số bằng 1 và detector đóng băng. Không cộng thêm một CE nền độc lập nữa. Báo cáo đã đối chiếu trực tiếp `train.py` thay vì chép công thức dự kiến từ một phiên bản cũ.

## 16. Vì sao cần loss phụ geometry?

Nhánh mới cần một tín hiệu giám sát trực tiếp để học chọn ID thay vì chỉ nhận gradient qua hệ số residual. NEW và target không có lag hợp lệ được loại khỏi nhóm query tính CE geometry. Điều này không có nghĩa mọi query đều có geometry đáng tin cậy.

## 17. Có thay đổi calibration theo seed không?

Không, nhóm run này dùng Tb=2, Tg=0,5, lambda=1 cố định. Các giá trị xuất phát từ quá trình nghiên cứu trước đó, không được chọn lại theo metric từng seed của nhóm bảng hiện tại. Không vì vậy mà gọi toàn bộ validation là chưa từng được quan sát trong quá trình phát triển.

## 18. Detector đóng băng sao DetA vẫn tăng?

Phát hiện ban đầu từ detector giống chính sách, nhưng đầu ra cuối phụ thuộc quyết định NEW, bộ lọc newborn và vòng đời track. Association khác đi có thể làm tập hộp xuất ra khác đi. Không quy mức tăng DetA này thành detector được học thêm trong các run đóng băng.

## 19. Cùng 10 epoch có phải so sánh công bằng hoàn toàn?

Trong một ngân sách, MOTIP nền/MOTIP + G4 dùng cùng cấu hình huấn luyện chính. Giữa các ngân sách, dataset lớn cho nhiều update hơn, còn MOTIP + G4 có bước pretrain MLP bổ sung. Vì vậy chưa thể tách hoàn toàn lợi ích kiến trúc, dữ liệu và tổng compute; cần một control cân bằng update/compute để kết luận mạnh hơn.

## 20. Tại sao không dùng confusion matrix cho 50 ID?

ID là ký hiệu tương đối thay đổi theo clip/slot, không phải 50 lớp người cố định của một bài phân loại thông thường. Gộp một confusion matrix toàn dataset theo số slot sẽ gây hiểu nhầm. Metric tracking chuẩn HOTA, AssA, IDF1, IDSW phù hợp hơn; nếu nghiên cứu query-level thì cần định nghĩa candidate set và NEW rõ ràng.

## 21. IDF1 và IDSW khác nhau thế nào?

IDF1 đánh giá độ đúng ID ở mức quỹ đạo sau ghép danh tính toàn cục. IDSW là đếm sự kiện thay đổi theo CLEAR matching ở chuỗi thời gian. Ít switch không bảo đảm tốt nếu tracker mất dấu hoặc gán ID không bao phủ trọn quỹ đạo. Xem cả AssA, IDF1 và detection.

## 22. HOTA có bằng căn DetA nhân AssA trong bảng không?

Ở một ngưỡng IoU thì HOTA_alpha = sqrt(DetA_alpha × AssA_alpha). Nhưng giá trị chính thức tổng hợp nhiều ngưỡng, nên không lấy căn hai giá trị trung bình đã hiển thị rồi đòi phải đúng hoàn toàn bằng HOTA trong bảng.

## 23. Kết quả chính là checkpoint cuối hay best?

Nhóm bảng chính là checkpoint tốt nhất theo AssA trên small validation cố định 3.894 frame rồi đánh giá trên full validation. Không gọi nó là final-vs-final. Bảng từng seed ghi rõ epoch được chọn. Các bảng best theo full validation cũ khác quy tắc không được trộn vào bảng chính.

## 24. Ba seed có đủ để khẳng định thống kê không?

Chưa đủ để nói chắc một hiệu ứng phổ quát. Báo cáo dùng mean ± sample std, ddof=1, và chênh lệch ghép cặp theo seed. Có thể nói cả ba seed trong từng ngân sách đều cùng chiều, không gọi chín cặp lồng nhau là chín thử nghiệm độc lập hoàn toàn.

## 25. Tại sao không so trực tiếp với điểm SOTA của paper?

Paper dùng ngân sách, detector/training recipe và split test khác. Đồ án dùng subset train và full validation, detector đóng băng. So trực tiếp hai con số mà không khớp protocol sẽ không đánh giá đúng phần cải tiến.

## 26. Đâu là phần tự thực hiện?

Phương pháp nền, detector và đánh giá dùng mã nguồn tác giả có trích dẫn/giấy phép. Phần bổ sung là biểu diễn G4 theo cặp, scorer, giám sát/fusion, kiểm tra alignment và thiết kế/thu thập thực nghiệm; cùng các script tích hợp, tổng hợp, demo và báo cáo. Không nhận việc phát minh MOTIP hay viết toàn bộ từ đầu là thành tích của đồ án.

## 27. Có vừa huấn luyện lại các run trong đêm chuẩn bị không?

Không. Theo lựa chọn của người thực hiện, đồ án tái sử dụng các run đã có và kiểm tra lại summary/protocol. Chạy mới phần suy luận demo; dự đoán được đối chiếu lại với artifact cũ. Sự khác nhau giữa kết quả tái sử dụng và lần chạy mới đã ghi trong protocol.

## 28. Video có dùng ground truth để gán ID hoặc làm đẹp không?

Không. Script suy luận tạo dataset với `load_annotation=False`; model nhận ảnh và trạng thái dự đoán. Màu và vệt được vẽ từ ID/box dự đoán. Video là hình trực quan dựng từ suy luận thực tế, không phải annotation được dùng làm dự đoán và không phải FPS màn hình được hiểu là throughput mô hình.

## 29. Video chạy 20 FPS sao throughput khoảng 9 FPS?

20 FPS là tốc độ phát lại của dữ liệu/video đã mã hóa. Throughput suy luận được đo riêng sau warm-up, có CUDA synchronization. Video hoàn tất có thể phát ở tốc độ gốc dù quá trình tính toán từng frame chậm hơn.

## 30. Đâu là bước tiếp theo có ý nghĩa nhất?

Tách tập chọn model khỏi một test sequence-level thật sự độc lập, cân bằng số update và geometry pretrain compute, rồi đánh giá full-data hoặc thêm dataset khác. Trước đó chưa cần làm geometry encoder phức tạp hơn hoặc tuyên bố SOTA phổ quát.

## 31. GT-linked Pred–Pred có dùng hộp GT thay hộp detector không?

Không. Sau khi detector dự đoán, hộp được ghép với GT train để biết thuộc người nào. Tọa độ vẫn là tọa độ detector. GT chỉ giúp nối lịch sử và tạo nhãn cùng/khác đối tượng; mã ID không nằm trong vector G4. Khi suy luận thật không có GT hỗ trợ bước này.

## 32. Hai tình huống minh họa đã được kiểm chứng thế nào?

Đã đối chiếu GT và cả chín frame quanh mỗi sự kiện bằng phép ghép CLEAR/TrackEval. Người GT 2 trong dancetrack0004 bị MOTIP nền đổi từ ID 0 sang 2, còn MOTIP + G4 giữ 0. Người GT 8 trong dancetrack0034 bị đổi từ 387 sang 442, còn nhánh có G4 giữ 6. Đây là các ví dụ tích cực chọn có chủ đích, không phải mẫu ngẫu nhiên hay ước lượng tỷ lệ lỗi. GT chỉ được đọc để phân tích sau suy luận.

## Câu chốt khi cần trả lời ngắn

“Đồ án tái hiện MOTIP bằng mã nguồn có trích dẫn và khảo sát một nhánh geometry nhỏ. Trên các subset DanceTrack và ba seed đã kiểm chứng, nhánh này cải thiện liên kết nhất quán. Chúng em công khai tiền huấn luyện, cách chọn checkpoint và giới hạn validation, nên không đồng nhất kết quả này với việc vượt paper gốc trên benchmark test.”
