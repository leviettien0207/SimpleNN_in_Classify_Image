# SimpleNN_in_Classify_Image
![image](https://user-images.githubusercontent.com/96035211/184624807-649cbcdd-7d7f-4f23-aa22-2f1aad445bcb.png)

Những thông số chúng ta sửa sẽ liên quan tới ảnh trên:
- số hidden layer (sau layer Flatten trong code và trước layer Dense 10 cuối cùng trong code): <br>
   + số neuron (1 hình tròn là 1 neuron ở như ở hình minh họa trên)
   + activation function: để khử tuyến tính, chứ nếu giá trị cứ qua một layer gấp 10 lên thì tới layer 15 là model quá tải
- Trong ảnh là các neuron đang được kết nối hoàn toàn (một neuron được nối từ toàn bộ neuron của layer trước), điều này dấn tới khả năng neuron học lỏm của nhau đưa ra giá trị tương đồng -> mất hiệu quả. Sử dụng layers.dropout(số lượng kết nối sẽ bị mất) <br>

Lưu ý:
- tf.keras.model.sequential([]) params là một list, và các layer sẽ được tạo ra theo list
- Các layers trước Flatten() ở trong code chính là các tấm lọc của mạng tích chập (Convolutional Neural Network), phần này khó hơn một tí, cần đánh giá dataset hiệu quả để có các filter tấm lọc hiệu quả

VỀ DỮ LIỆU<br>
Một trong những nguồn dữ liệu (dataset) lớn nhất là kaggle.com.<br>
Trên kaggle.com có rất nhiều dữ liệu đa dạng và kaggle có hỗ trợ api, command để lấy dữ liệu thay vì tải về.<br>
<br>
Chúng ta có 2 cách để chuẩn bị dataset từ kaggle:
1. Vào link để tải
2. Sử dụng api dược kaggle cung cấp cá nhân và sử dụng command/code để tải

Về cách sử dụng tính năng api, command thì cần tạo tài khoản kaggle.com<br>
****
Tạo tài khoản kaggle > Account > API > Create new API token. <br>
Tải xuống và lưu trữ ở địa chỉ sau: C:\Users\xxxx\.kaggle <br>
Command "kaggle d list" để kiểm tra xem API đã hoạt động hay chưa
****
Dữ liệu tôi sẽ sử dụng ở bài này: <br>
https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist?select=dataset <br>
****
Về cấu trúc file tập dataset
1. training <br>
   0 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>
   1 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>
   ... <br>
   9 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>
2. validation
   0 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>
   1 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>
   ... <br>
   9 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.png <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10772.png <br>

LƯU Ý: Để có nhãn (label, kết quả ảnh) tự động, việc cấu trúc dataset nghiêm khắc đúng format (tên folder sẽ chính là nhãn, ví dụ 0, 1, 2,...) <br>
****
Tham khảo activation function: https://www.mygreatlearning.com/blog/activation-functions/ <br>

chạy command line <br>
cd SimpleNN_in_Classify_Image
py app.py
