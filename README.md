# SimpleNN_in_Classify_Image

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
