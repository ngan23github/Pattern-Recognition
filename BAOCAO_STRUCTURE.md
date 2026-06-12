# ĐỀ CƯƠNG CHI TIẾT BÁO CÁO MÔN HỌC
**Môn học:** Phân tích Nhận dạng mẫu
**Đề tài:** Ứng dụng Deep Learning và Pipeline nhận diện trái cây trong môi trường thực tế

---

## 1. Phát biểu bài toán
*   **Bối cảnh:** Nhận diện và phân loại trái cây tự động có vai trò quan trọng trong nông nghiệp thông minh, hệ thống thanh toán tự động tại siêu thị (self-checkout), và quản lý kho hàng.
*   **Vấn đề/Thách thức:** 
    *   Hầu hết các bộ dữ liệu có sẵn (như Fruit 360) được chụp trong điều kiện phòng thí nghiệm với phông nền trắng, ánh sáng hoàn hảo.
    *   Tuy nhiên, khi đưa ứng dụng ra thực tế, hình ảnh thu thập được có **background phức tạp**, **khoảng cách chụp khác nhau**, và **góc độ đa dạng**. Điều này gây sụt giảm độ chính xác (accuracy) nghiêm trọng do sự khác biệt phân phối dữ liệu (domain shift).
*   **Mục tiêu đề tài:** Xây dựng một hệ thống hoàn chỉnh từ việc phân tích, huấn luyện các mô hình phân loại (Classification), cho đến thiết kế một **Pipeline Inference thực tế** có khả năng xử lý ảnh nhiễu, tích hợp vào hệ thống Web API phục vụ người dùng.

---

## 2. Thiết kế giải pháp

### 2.1. Chọn dataset
*   **Dataset sử dụng:** [Fruit 360](https://www.kaggle.com/datasets/moltean/fruits) (đã được tiền xử lý/gộp lại).
*   **Đặc điểm:** Bao gồm hàng chục ngàn bức ảnh, độ phân giải 100x100 pixel, trên nền trắng hoàn toàn (White background).
*   **Lý do chọn:** Dữ liệu chuẩn, đa dạng lớp (class), thích hợp để so sánh khả năng trích xuất đặc trưng của các mạng CNN từ cơ bản đến nâng cao.

### 2.2. Thiết kế xây dựng mô hình (Pipeline)
Để giải quyết bài toán chênh lệch dữ liệu (từ nền trắng sang nền thực tế), giải pháp đề xuất một Pipeline Inference gồm các bước liên hoàn:
1.  **Object Detection (YOLOv8):** Nhận ảnh đầu vào, định vị (Bounding Box) và cắt (Crop) chính xác vùng chứa trái cây để loại bỏ bớt bối cảnh dư thừa.
2.  **Background Removal (Rembg):** Xóa phông nền phức tạp của ảnh thực tế, chuyển về phông nền trong suốt/trắng, ép ảnh thực tế trở nên "giống" với dữ liệu Fruit 360 nhất có thể.
3.  **Letterbox & Resize:** Giữ nguyên tỷ lệ khung hình (Aspect Ratio), đệm viền (padding) và thay đổi kích thước chuẩn (ví dụ 100x100 hoặc 224x224) để đưa vào mô hình mà không làm méo mó hình dáng trái cây.
4.  **Classification (Deep Learning Models):** Đưa ảnh đã xử lý vào mô hình (CNN/ResNet/EfficientNet) để phân loại và lấy top kết quả dự đoán.

### 2.3. Công nghệ sử dụng
*   **Xây dựng và huấn luyện mô hình (AI Framework):** PyTorch, Torchvision.
*   **Tiền xử lý và Computer Vision:** OpenCV, Rembg (Alpha Matting), Ultralytics YOLOv8.
*   **Trực quan hóa:** Matplotlib, Seaborn.
*   **Backend API:** FastAPI, Uvicorn (nhanh, hỗ trợ bất đồng bộ, lý tưởng cho Model as a Service).
*   **Frontend UI:** Streamlit (xây dựng giao diện web tương tác nhanh bằng Python).

---

## 3. Kết quả các thử nghiệm xây dựng mô hình (3 thử nghiệm)
*Giới thiệu: Để chọn ra mô hình tốt nhất đưa vào sản phẩm, nhóm đã tiến hành 3 thử nghiệm trên các kiến trúc từ cơ bản đến hiện đại.*

### 3.1. Thử nghiệm 1: Custom CNN (Baseline Model)
*   **Kiến trúc:** Xây dựng từ đầu một mạng Convolutional Neural Network (CNN) thuần túy gồm 4 block. Mỗi block chứa: `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`. Lớp cuối là Fully Connected kết hợp Dropout.
*   **Mục đích:** Tạo ra một baseline cơ sở để làm hệ quy chiếu đánh giá.
*   **Kết quả:** Độ chính xác ở mức khá trên test set nhưng dễ bị nhiễu. Số lượng tham số rất nhỏ (~1.2M), tốn ít tài nguyên nhưng không đủ sâu để trích xuất các đặc trưng phức tạp.

### 3.2. Thử nghiệm 2: ResNet-18 (Transfer Learning)
*   **Kiến trúc:** Sử dụng kiến trúc mạng phần dư (Residual Network - Skip Connections) để giải quyết vấn đề Vanishing Gradient.
*   **Kỹ thuật:** Áp dụng **Transfer Learning** (weights pretrained trên ImageNet) kết hợp **Differential Learning Rate** (Tốc độ học của Classifier Head lớn hơn gấp 10 lần so với Backbone để giữ lại các đặc trưng tổng quát đã học được).
*   **Kết quả:** Độ chính xác tăng vọt (có thể lấy số liệu % từ notebook). Tuy nhiên, số lượng tham số lớn (~11.3M) và khối lượng tính toán cao (~1.8 Tỉ FLOPs).

### 3.3. Thử nghiệm 3: EfficientNet-B0 (Transfer Learning)
*   **Kiến trúc:** Dòng kiến trúc SOTA tối ưu hóa hiệu năng, sử dụng MBConv và SE block (Squeeze-and-Excitation). 
*   **Kỹ thuật:** Transfer learning, tận dụng lợi thế của **Compound Scaling** (mở rộng đồng đều cả chiều sâu, chiều rộng mạng và độ phân giải ảnh).
*   **Kết quả:** Hiệu suất phân loại rất cao trên tập Test, tương đương hoặc nhỉnh hơn ResNet-18 nhưng cực kỳ mỏng nhẹ.

### 3.4. Đánh giá 3 thử nghiệm và Quyết định chọn Model
Dựa trên các bảng biểu phân tích từ Notebook, nhóm quyết định **chọn EfficientNet-B0 để làm sản phẩm cuối cùng** vì 4 ưu điểm vượt trội:
1.  **Compound Scaling:** Khả năng nhận diện tốt ở nhiều khoảng cách và kích thước trái cây nhờ được scale đồng đều mọi chiều.
2.  **SE block (Squeeze-and-Excitation):** Cơ chế attention kênh màu giúp tự động lọc bỏ nền nhiễu, tập trung vào đặc điểm của quả.
3.  **Khái quát hóa tốt (Tránh Overfitting):** Với chỉ ~4.3M tham số (bằng 1/3 ResNet-18), mô hình ít bị quá khớp (overfit) vào phông nền trắng của tập huấn luyện, giúp dự đoán ảnh thực tế ổn định hơn.
4.  **Hiệu năng tính toán (Tốc độ):** Chỉ tiêu tốn 0.39 Tỉ FLOPs (so với 1.8 Tỉ của ResNet-18), giúp suy luận qua Backend cực nhanh, tối ưu hóa khi chạy trên thiết bị edge/điện thoại/camera.

---

## 4. Kết quả của sản phẩm

### 4.1. Màn hình ứng dụng web (Frontend - Streamlit)
*   *(Trong báo cáo, bạn hãy chèn ảnh chụp màn hình ứng dụng Streamlit)*.
*   **Giao diện trực quan:** Cho phép người dùng Upload hoặc Kéo-Thả (Drag & Drop) ảnh thực tế từ máy tính.
*   **Hiển thị trung gian:** Có thể hiển thị quá trình pipeline hoạt động (ví dụ: Hình ảnh sau khi đã bị YOLO cắt Bounding Box và Rembg xóa phông).
*   **Hiển thị kết quả:** Hiển thị tên trái cây có xác suất cao nhất kèm theo biểu đồ Top-5 dự đoán (Confidence Scores). Đóng khung Bounding Box lên ảnh gốc.

### 4.2. Hệ thống API Backend và Endpoints (FastAPI)
*   Xây dựng kiến trúc tách biệt (Decoupled Architecture), Backend đóng vai trò như một Microservice phục vụ AI (Model-as-a-Service).
*   **Endpoint chính (`POST /predict`):** 
    *   **Input:** Nhận file ảnh định dạng `multipart/form-data`.
    *   **Processing:** Chạy qua toàn bộ Pipeline (YOLO -> Rembg -> EfficientNet).
    *   **Output:** Trả về định dạng JSON, ví dụ: 
        ```json
        {
           "success": true,
           "predictions": [
               {"class_name": "Apple Braeburn", "confidence": 0.98},
               {"class_name": "Apple Crimson Snow", "confidence": 0.01}
           ],
           "box_coordinates": [xmin, ymin, xmax, ymax]
        }
        ```
*   *(Trong báo cáo, bạn hãy chèn ảnh chụp màn hình test API bằng công cụ Postman)*.

---

## 5. Hạn chế và hướng phát triển

### 5.1. Hạn chế hiện tại
*   **Giới hạn của việc tách phông nền (Rembg):** Thuật toán U-2-Net bên trong Rembg đôi khi xóa nhầm các chi tiết của trái cây nếu màu sắc của quả và phông nền phía sau quá giống nhau (ít độ tương phản).
*   **Che khuất (Occlusion):** Nếu hình chụp có nhiều quả chồng chéo lên nhau hoặc bị tay người cầm che khuất một phần lớn, mô hình Classification có thể bị giảm độ tin cậy.
*   **Nhiễu từ ảnh mạng (YOLOv8 nano):** YOLOv8 bản nano tuy nhanh nhưng thỉnh thoảng có thể bắt trượt khung hình (Bounding Box) nếu môi trường ánh sáng quá phức tạp.

### 5.2. Hướng phát triển trong tương lai
*   **Cải thiện Dataset:** Thu thập thêm dữ liệu hình ảnh trái cây trong siêu thị, trên cây, hoặc trong giỏ (ảnh có background thật) để fine-tune trực tiếp mô hình thay vì chỉ phụ thuộc vào thuật toán xóa phông nền.
*   **Nâng cấp mô hình (End-to-End Detection):** Khi có đủ dữ liệu thực tế được gán nhãn Bounding Box, có thể huấn luyện trực tiếp họ mô hình Object Detection (như YOLOv10 hoặc RT-DETR) để vừa phát hiện vừa phân loại cùng lúc, bỏ qua bước trung gian tách phông.
*   **Triển khai Cloud (Deployment):** Đóng gói ứng dụng bằng Docker và triển khai lên các dịch vụ Cloud như AWS EC2, Google Cloud Run hoặc Render để người dùng có thể truy cập qua Internet từ bất kỳ đâu. Tích hợp thêm tính năng chụp ảnh trực tiếp từ Camera điện thoại trên giao diện Web.