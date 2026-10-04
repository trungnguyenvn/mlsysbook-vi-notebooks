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

## Nguồn và giấy phép

Nội dung gốc: [*Machine Learning Systems*](https://mlsysbook.ai/) của Vijay Janapa Reddi và
cộng sự, MIT Press, © 2024–2026 Harvard University, giấy phép
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Sổ tay nào phỏng theo một
phòng thí nghiệm (`labs/vol1/…`) của sách thì ghi rõ ở ô cuối của nó; các phòng thí nghiệm đó
© 2026 President and Fellows of Harvard College, CC BY-NC-SA 4.0.

Các sổ tay này là tác phẩm phái sinh, dùng cùng giấy phép (toàn văn ở [LICENSE](LICENSE)):
ghi công, chia sẻ tương tự, **phi thương mại**. Đây là bản tiếng Việt không chính thức; các tác
giả và nhà xuất bản của sách không tham gia và không bảo trợ nó.
