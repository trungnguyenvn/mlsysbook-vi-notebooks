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
