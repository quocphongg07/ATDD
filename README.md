**Phát hiện phần mềm độc hại Android bằng GCN**

Dự án này là quá trình tái thực nghiệm phương pháp được trình bày trong bài báo khoa học: "An Android Malware Detection Approach to Enhance Node Feature Differences in a Function Call Graph Based on GCNs" (Wu et al., 2023).
Mục tiêu cốt lõi của phương pháp này là sử dụng Mạng Tích chập Đồ thị (GCN) trên các Đồ thị Lời gọi Hàm (FCG) được trích xuất từ các tệp APK. Đóng góp chính của bài báo là đề xuất một đặc trưng mới (Function Weight) và một phương pháp trích xuất đồ thị con (S-FCSG) để tăng cường độ chính xác của mô hình.

**1. Yêu cầu dữ liệu**

Kho lưu trữ này chỉ chứa mã nguồn (scripts) để xử lý và huấn luyện. Do các hạn chế về bảo mật và kích thước, các tập dữ liệu được sử dụng trong giai đoạn này không thể tải lên đây.
Để chạy dự án này, người dùng phải tự thu thập các tài nguyên sau:
- Các tệp .apk Thô: Người dùng cần một bộ dữ liệu lớn gồm các tệp .apk lành tính và độc hại. Bài báo gốc sử dụng:
+ Độc hại: Drebin, CICMalDroid 2020.
+ Lành tính: Androzoo.
(Trong quá trình thực nghiệm, chúng ta đã sử dụng CIC-AndMal2017 và các mẫu từ APKMirror).
- Bản đồ API (aosp_map.json): Đây là một tệp JSON người dùng phải tự xây dựng, có tác dụng ánh xạ tên hàm API (định dạng Dalvik) tới mức độ rủi ro (1: nguy hiểm, 0: bình thường).
- Sử dụng dữ liệu từ PScout (được đề cập trong bài báo) làm nguồn tra cứu để tạo tệp này.
- 
**2. Cài đặt Môi trường**
  
Quy trình này cực kỳ nhạy cảm với phiên bản thư viện. Việc cài đặt các gói C++ (PyTorch, DGL) trên Kali Linux hoặc bằng pip tiêu chuẩn rất dễ gặp lỗi (iJIT_NotifyEvent, libgraphbolt.so, ModuleNotFoundError). Cách làm ổn định duy nhất và được khuyến nghị là sử dụng Miniconda với Python 3.11.
Quy trình được chia thành hai môi trường riêng biệt:
- Môi trường xử lý (Processing): Dùng để chạy Giai đoạn 1-4 (an toàn trên sandbox như Kali).
- Môi trường huấn luyện (Training): Dùng để chạy Giai đoạn 5 (trên máy thật Windows).
# 2.1. Môi trường xử lý (Dùng trên Kali)

Môi trường này chỉ cài đặt các gói cần thiết để phân tích .apk và xử lý đồ thị.
- Tạo môi trường conda mới:
conda create -n processing_env python=3.11
- Kích hoạt môi trường:
conda activate processing_env
- Cài đặt các gói xử lý (dùng pip):
pip install androguard joblib pandas pyarrow networkx

# 2.2 Môi trường huấn luyện (Dùng trên máy thật)

Môi trường này cài đặt các gói học sâu (PyTorch & DGL) một cách ổn định.
- Tạo môi trường conda mới:
conda create -n training_env python=3.11
- Kích hoạt môi trường:
conda activate training_env
- Cài đặt PyTorch và DGL:
conda install pytorch dgl -c pytorch -c dglteam
- Cài đặt các gói phụ trợ:
pip install numpy
