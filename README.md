# Sổ tay thực hành: Hệ thống Học máy

*Colab notebooks (in Vietnamese) for the chapters of* Machine Learning Systems *(Reddi et al.,
MIT Press), companions to the Vietnamese video lessons at https://d3lu8vk4we0dc.cloudfront.net/.*

Mỗi chương của bài giảng tiếng Việt [Hệ thống Học máy](https://d3lu8vk4we0dc.cloudfront.net/) có một đến bốn sổ tay. Mỗi sổ
tay lấy một ý của chương rồi tự tính lại bằng code: đếm, đo, vẽ, rồi đặt kết quả cạnh con số
của sách.

**Cách dùng.** Bấm nút *Open in Colab* ở một dòng bên dưới, rồi chạy ô *Chuẩn bị* trước. Mọi sổ
tay chạy trên CPU (Runtime › Change runtime type › CPU) và không cài thêm thư viện nào: những gì
chúng dùng đều có sẵn trên Colab. Dòng `✓` nghĩa là con số tính được khớp với sách; khi bạn đổi
tham số ở phần *Thử thêm*, dấu `✗` là điều được chờ đợi.

Các tệp `.ipynb` ở đây được sinh tự động và không kèm kết quả chạy.

## [Tính toán nơ-ron](https://d3lu8vk4we0dc.cloudfront.net/lessons/vol1-ch05-nn-computation/)

*Machine Learning Systems — Tập I, Chương 5*

| | Sổ tay | Nội dung |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch05-nn-computation/01-lan-truyen-nguoc.ipynb) | [Lan truyền ngược bằng tay](vol1-ch05-nn-computation/01-lan-truyen-nguoc.ipynb) | Lần theo gradient qua mạng tí hon của chương bằng NumPy, đối chiếu với autograd và sai phân hữu hạn, rồi đếm tham số, MAC và bộ nhớ của mạng 784-128-64-10. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch05-nn-computation/02-so-hoc-va-ham-kich-hoat.ipynb) | [Số học không nhánh vẫn có rủi ro](vol1-ch05-nn-computation/02-so-hoc-va-ham-kich-hoat.ipynb) | Lặp lại phép đổi số của Ariane 5, đo miền giá trị của FP32 và FP16, xem một NaN lan khắp mô hình, rồi tính sigmoid, tanh, ReLU, softmax và entropy chéo bằng autograd, cạnh thuế transistor và thời gian đo trên CPU. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch05-nn-computation/03-ba-loi-tinh-tren-mot-chu-so.ipynb) | [Ba lối tính trên một chữ số](vol1-ch05-nn-computation/03-ba-loi-tinh-tren-mot-chu-so.ipynb) | Dựng một bộ quy tắc 100 phép so sánh, tự tính HOG 441 đặc trưng cho một SVM tuyến tính và huấn luyện mạng 784-128-64-10, đếm phép của từng lối cạnh độ chính xác đo được, rồi thử mốc đơn giản và lợi ích giảm dần khi nới mạng. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch05-nn-computation/04-huan-luyen-va-suy-luan.ipynb) | [Từ vòng lặp huấn luyện đến đường ống suy luận](vol1-ch05-nn-computation/04-huan-luyen-va-suy-luan.ipynb) | Đếm phép một lượt truyền xuôi và nhẩm thời gian một epoch, viết vòng lặp minibatch với bốn cấu hình batch và tốc độ học, giữ checkpoint khi quá khớp, rồi dựng đường ống suy luận: bộ nhớ, độ trễ theo từng bước và theo batch, ngưỡng từ chối theo chi phí, và một dịch chuyển phân phối đọc theo D·A·M. |

## [Kiến trúc mạng](https://d3lu8vk4we0dc.cloudfront.net/lessons/vol1-ch06-nn-architectures/)

*Machine Learning Systems — Tập I, Chương 6*

| | Sổ tay | Nội dung |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch06-nn-architectures/01-gia-cua-mot-gia-dinh.ipynb) | [Giá của một giả định](vol1-ch06-nn-architectures/01-gia-cua-mot-gia-dinh.ipynb) | Đếm tham số, phép nhân cộng dồn và byte của tầng dày đặc, tích chập và tích chập tách theo chiều sâu, đối chiếu với numel() của PyTorch, rồi đếm ResNet-50 và MobileNetV2 thật của torchvision: chữ ký tải công việc, phổ mật độ tính toán, mật độ của từng tầng và tỉ số FLOP so với tỉ số thời gian đo trên CPU. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch06-nn-architectures/02-tich-chap-tu-ben-trong.ipynb) | [Tích chập từ bên trong](vol1-ch06-nn-architectures/02-tich-chap-tu-ben-trong.ipynb) | Viết tích chập bằng bảy vòng lặp rồi bằng im2col, đếm dữ liệu im2col nhân bản và đo ba cách hiện thực trên tầng ImageNet của chương, dựng Winograd F(2×2, 3×3) bằng tay, kiểm lại Ví dụ 1.4 về đẳng biến và chỗ nó vỡ, đếm kết nối cục bộ và chia sẻ tham số, rồi đo thiên kiến quy nạp: MLP và hai CNN nhỏ trên chữ số MNIST bị dịch. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch06-nn-architectures/03-tuan-tu-va-song-song.ipynb) | [Tuần tự và song song](vol1-ch06-nn-architectures/03-tuan-tu-va-song-song.ipynb) | Viết RNN của chương bằng tay và đo đường tới hạn của nó theo độ dài chuỗi và batch, tính chính xác ma trận điểm tập trung (33.6 MB ở 4,096 token, 240 GB và 7,680 GB của Napkin Math 1.1) và đo nó, viết tập trung chia khối với softmax trực tuyến, dựng một khối transformer bằng tay, rồi tính bức tường băng thông khi sinh token và bộ nhớ đệm KV. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch06-nn-architectures/04-tu-khoi-dung-chung-toi-quyet-dinh.ipynb) | [Từ khối dùng chung tới một quyết định kiến trúc](vol1-ch06-nn-architectures/04-tu-khoi-dung-chung-toi-quyet-dinh.ipynb) | So chuẩn hóa theo batch với chuẩn hóa theo tầng, kiểm Định lý 1.2 về kết nối bỏ qua, tính năng lượng của một tầng theo 3.7 pJ và 640 pJ, tính bức tường dung lượng của bảng embedding và đo một lần tra bảng, dựng bảng độ phức tạp và đặt ba chữ ký vào định luật sắt trên một thiết bị giả định, rồi chọn kiến trúc cho máy bẫy ảnh bằng một bảng khả thi. |

## [Huấn luyện mô hình — khi định luật sắt trở thành công cụ hằng ngày](https://d3lu8vk4we0dc.cloudfront.net/lessons/vol1-ch08-model-training/)

*Machine Learning Systems — Tập I, Chương 8*

| | Sổ tay | Nội dung |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch08-model-training/01-dinh-luat-sat.ipynb) | [Định luật sắt của huấn luyện: đếm phép toán, đo mức sử dụng, tính hoá đơn](vol1-ch08-model-training/01-dinh-luat-sat.ipynb) | Đếm FLOP một tầng GPT-2 tới cùng, đếm vì sao lượt xuôi cộng ngược gấp 3 lượt xuôi, đo thông lượng đỉnh và mức sử dụng của CPU theo kích thước batch, rồi tính thuê hay mua cụm cho Llama 2 70B, MFU và năng lượng một lần chạy. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch08-model-training/02-bo-toi-uu-va-batch.ipynb) | [Bộ tối ưu và batch: quỹ đạo, trạng thái, lịch tốc độ học và tích luỹ gradient](vol1-ch08-model-training/02-bo-toi-uu-va-batch.ipynb) | Thả SGD, động lượng, RMSprop và Adam lên một mặt mất mát, đếm trạng thái mà từng bộ tối ưu của PyTorch giữ, đo nhiễu gradient theo 1/√B, vẽ lịch tốc độ học của GPT-2 XL, rồi kiểm rằng tích luỹ gradient cho đúng gradient của batch lớn và tìm hai cách làm nó sai. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch08-model-training/03-bo-nho-huan-luyen.ipynb) | [Bộ nhớ huấn luyện: bốn số hạng, độ chính xác hỗn hợp và lưu điểm](vol1-ch08-model-training/03-bo-nho-huan-luyen.ipynb) | Đếm bằng hook bốn số hạng của phương trình bộ nhớ huấn luyện, nhân công thức 16 byte mỗi tham số lên 7, 20 và 175 tỷ tham số, đếm hệ số 5 byte của ma trận điểm attention và dựng lại bảng giá trị kích hoạt của GPT-2, đọc FP32, FP16, BF16 từ PyTorch và xem gradient thật rơi khỏi miền FP16, rồi đếm bộ nhớ và số lần tính lại của torch.utils.checkpoint và đi lại phía bộ nhớ của ca GPT-2 trên một V100. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch08-model-training/04-di-chuyen-du-lieu.ipynb) | [Di chuyển dữ liệu: quy trình nạp, nạp trước, FlashAttention và AllReduce](vol1-ch08-model-training/04-di-chuyen-du-lieu.ipynb) | Áp quy tắc chặng chậm nhất cho quy trình dữ liệu và lần theo quy trình GPT-2 của chương từ đĩa tới PCIe, đo nạp trước bằng một luồng nền và đo vì sao GIL chặn luồng Python, viết attention chia khối với softmax trực tuyến và kiểm nó khớp attention chuẩn, rồi mô phỏng AllReduce vòng, đếm byte mỗi GPU gửi và tính tường mạng của chương. |

## [Nén mô hình — đổi phần dư thừa lấy ràng buộc triển khai](https://d3lu8vk4we0dc.cloudfront.net/lessons/vol1-ch10-model-compression/)

*Machine Learning Systems — Tập I, Chương 10*

| | Sổ tay | Nội dung |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch10-model-compression/01-cat-tia-va-do-thua.ipynb) | [Cắt tỉa: số 0 nào làm mô hình nhỏ lại, số 0 nào làm nó nhanh hơn](vol1-ch10-model-compression/01-cat-tia-va-do-thua.ipynb) | Cắt tỉa theo độ lớn một mạng MNIST nhỏ, toàn cục và theo từng tầng, có cấu trúc và không cấu trúc, rồi đo thời gian phép nhân và đếm byte của từng cách lưu số 0 để tìm điểm hoà vốn. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch10-model-compression/02-chung-cat-va-hang-thap.ipynb) | [Chưng cất tri thức và phân rã hạng thấp](vol1-ch10-model-compression/02-chung-cat-va-hang-thap.ipynb) | Đọc tri thức ẩn trong nhãn mềm ở nhiều nhiệt độ, so mô hình trò tự học với mô hình trò được chưng cất và tìm khi nào chưng cất không giúp, rồi phân rã một tầng đã huấn luyện bằng SVD và đo ví dụ ma trận 4096 nhân 4096 ở hạng 128. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch10-model-compression/03-luong-tu-hoa-bang-tay.ipynb) | [Lượng tử hoá bằng tay: hệ số co giãn, điểm không, và cái giá của từng bit](vol1-ch10-model-compression/03-luong-tu-hoa-bang-tay.ipynb) | Tính lại hệ số co giãn và điểm không của chương, so đối xứng với bất đối xứng, theo tầng với theo kênh, đo sai số theo số bit và tác hại của một điểm ngoại lai, rồi lượng tử hoá một mạng MNIST, đếm đúng từng byte và thử QAT ở 2 bit. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch10-model-compression/04-byte-bang-thong-va-ti-so.ipynb) | [Byte, băng thông và tỉ số trên giấy: khi nào mô hình nén thật sự nhanh hơn](vol1-ch10-model-compression/04-byte-bang-thong-va-ti-so.ipynb) | Tính lại các phép kế toán byte của chương: mô hình nào vừa thiết bị nào, bộ nhớ đệm KV và token mỗi giây khi nghẽn ở bộ nhớ, thanh ghi SIMD, thang năng lượng, lưu lượng Conv-BN-ReLU trước và sau hợp nhất, định luật Amdahl; đo một tầng 4096 nhân 4096 trên CPU ở fp32, bf16 và int8 không có kernel int8, và theo kích thước batch. |

## [Tăng tốc phần cứng](https://d3lu8vk4we0dc.cloudfront.net/lessons/vol1-ch11-hw-acceleration/)

*Machine Learning Systems — Tập I, Chương 11*

| | Sổ tay | Nội dung |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch11-hw-acceleration/01-mo-hinh-roofline.ipynb) | [Mô hình roofline: hai trần, điểm gãy và chỗ đứng của từng phép toán](vol1-ch11-hw-acceleration/01-mo-hinh-roofline.ipynb) | Tính điểm gãy của V100, A100, H100, đếm FLOP và byte của các phép toán trong chương từ hình dạng tensor, đặt chúng lên roofline của A100, rồi xem batch, độ chính xác số và các ngộ nhận cuối chương làm dịch chuyển chúng. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch11-hw-acceleration/02-do-roofline-tren-cpu.ipynb) | [Đo roofline trên CPU của bạn, rồi hợp nhất kernel để bớt lưu lượng](vol1-ch11-hw-acceleration/02-do-roofline-tren-cpu.ipynb) | Đếm mật độ tính toán của phép nhân ma trận và phép theo từng phần tử, đo hai trần của máy đang chạy sổ tay, đặt một tầng dày đặc ở chín kích thước batch lên roofline đo được, rồi đếm và đo lưu lượng mà hợp nhất kernel bớt đi cho chuỗi ReLU, chuẩn hóa theo batch và affine. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch11-hw-acceleration/03-mang-systolic-va-chia-khoi.ipynb) | [Mảng systolic, chia khối và thứ tự vòng lặp: mỗi byte được dùng lại bao nhiêu lần](vol1-ch11-hw-acceleration/03-mang-systolic-va-chia-khoi.ipynb) | Tính lại ví dụ Tensor Core, mô phỏng từng chu kỳ một mảng systolic giữ trọng số tại chỗ cùng chi phí nạp đầy và xả, tính lại napkin math 1.1, đếm lưu lượng của phép nhân chia khối và khối nào vừa bộ nhớ nháp, rồi chạy cả 120 thứ tự vòng lặp của ví dụ 1.4 trên một tệp thanh ghi. |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trungnguyenvn/mlsysbook-vi-notebooks/blob/main/vol1-ch11-hw-acceleration/04-tu-kernel-toi-ca-he-thong.ipynb) | [Từ một kernel nhanh tới cả hệ thống: Amdahl, năng lượng, nhiều chip và cacbon](vol1-ch11-hw-acceleration/04-tu-kernel-toi-ca-he-thong.ipynb) | Dùng định luật Amdahl để xem vì sao 247 lần trên đường nhân ma trận chỉ còn 18.6 lần cho cả ứng dụng, tính năng lượng của việc di chuyển dữ liệu, lập lịch đệm kép, chạy thật một AllReduce vòng rồi tính kịch bản tám GPU và quy mô GPT-3, và tính lại napkin math 1.10 cùng độ nhạy của nó. |

## Nguồn và giấy phép

Nội dung gốc: [*Machine Learning Systems*](https://mlsysbook.ai/) của Vijay Janapa Reddi và
cộng sự, MIT Press, © 2024–2026 Harvard University, giấy phép
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Sổ tay nào phỏng theo một
phòng thí nghiệm (`labs/vol1/…`) của sách thì ghi rõ ở ô cuối của nó; các phòng thí nghiệm đó
© 2026 President and Fellows of Harvard College, CC BY-NC-SA 4.0.

Các sổ tay này là tác phẩm phái sinh, dùng cùng giấy phép (toàn văn ở [LICENSE](LICENSE)):
ghi công, chia sẻ tương tự, **phi thương mại**. Đây là bản tiếng Việt không chính thức; các tác
giả và nhà xuất bản của sách không tham gia và không bảo trợ nó.
