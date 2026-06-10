# 🍎 Nhận Diện Trái Cây — Pattern Recognition

Đồ án môn **Nhận Dạng Mẫu**, triển khai và so sánh các mô hình Deep Learning cho bài toán phân loại trái cây trên dataset **Fruit 360**, kết hợp pipeline nhận diện ảnh thực tế (Real-World Inference). Đặc biệt, hệ thống đã được triển khai thành ứng dụng Web hoàn chỉnh với API Backend (FastAPI) và giao diện Frontend (Streamlit).

---

## 📌 Tổng Quan

| Hạng mục | Chi tiết |
|---|---|
| Dataset | [Fruit 360](https://www.kaggle.com/datasets/moltean/fruits) |
| Framework | PyTorch |
| Mô hình so sánh | Custom CNN · ResNet-18 · EfficientNet-B0 |
| Inference thực tế | YOLOv8 (detect) + rembg (remove background) + Classifier |
| Triển khai Web | FastAPI (Backend) + Streamlit (Frontend) |

---

## 🗂️ Cấu Trúc Thư Mục

```
Pattern-Recognition/
│
├── fruit_recognition_3models.ipynb  # Notebook huấn luyện & đánh giá 3 model
│
├── models/                          # Checkpoint đã lưu sau training
│   ├── best_cnn.pth
│   ├── best_resnet18.pth
│   └── best_efficientnetB0.pth
│
├── backend/                         # Backend API code
│   ├── app.py                       # FastAPI server
│   └── requirements.txt             # Thư viện cho Backend
│
├── frontend/                        # Frontend UI code
│   ├── streamlit_app.py             # Ứng dụng Streamlit
│   └── requirements.txt             # Thư viện cho Frontend
│
└── data/
    ├── Training/
    ├── Test/
    └── test_real_multiple_fruits/   # Ảnh thực tế để test inference
```

---

## 📦 Dataset

Do dung lượng lớn, dataset không được lưu trực tiếp trong repo. Tải về theo hướng dẫn bên dưới:

| Dataset | Mô tả | Link |
|---|---|---|
| Fruit 360 | Data gốc, ~90k ảnh, nền trắng | [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) |
| Data đang dùng | Data đã được merged lại với nhau | [Google Drive](https://drive.google.com/file/d/1AtL0axJoFnEvIaZkjwetJH-BGMaSznqW/view?usp=drive_link) |

Sau khi tải về, giải nén và đặt vào đúng cấu trúc thư mục `data/` như mô tả ở trên.

---

## 🧠 Các Mô Hình

### 1. Custom CNN
Mạng CNN tự xây dựng từ đầu gồm 4 `ConvBlock` (Conv → BN → ReLU → MaxPool), dùng làm baseline để so sánh.

### 2. ResNet-18 (Transfer Learning)
Fine-tune ResNet-18 pre-trained trên ImageNet với **differential learning rate** — backbone lr thấp hơn FC head 10 lần để bảo toàn features đã học.

### 3. EfficientNet-B0 (Transfer Learning)
Fine-tune EfficientNet-B0 — kiến trúc dùng kỹ thuật **Compound Scaling** (đồng thời scale depth, width, resolution). Đạt accuracy cao hơn ResNet-18 với ít tham số hơn.

| Model | Tham số | Kiến trúc |
|---|---|---|
| Custom CNN | ~1.2M | Conv blocks |
| ResNet-18 | ~11M | Skip connection |
| EfficientNet-B0 | ~5.3M | MBConv + SE block |

---

## ⚙️ Cài Đặt và Khởi Chạy

### 1. Cài Đặt Môi Trường Cơ Bản

```bash
# Clone repo
git clone https://github.com/ngan23github/Pattern-Recognition.git
cd Pattern-Recognition

# Tạo môi trường ảo
python -m venv venv
source venv/Scripts/activate   # Git Bash (Windows)

# Cài các thư viện cần thiết cho Notebook
pip install -r requirements.txt
```

### 2. Chạy Notebook Huấn Luyện (Tuỳ chọn)
Mở và chạy `fruit_recognition_3models.ipynb`.  
- **Bỏ qua Training**: Code đã được thiết lập `TRAIN_FROM_SCRATCH = False`. Nếu folder `models/` đã có sẵn các file weight `.pth`, hệ thống sẽ tự nạp model và bỏ qua training để tiết kiệm thời gian. (Bạn có thể đổi thành `True` nếu muốn chạy lại từ đầu).
- Pipeline suy luận ảnh thực tế dùng YOLOv8 + Alpha Matting (chống lẹm) và Letterbox (chống méo hình).

### 3. Chạy Ứng Dụng Web (Model as a Service)

Chúng ta có 2 module độc lập: Backend (AI xử lý) và Frontend (Giao diện hiển thị).

**Bước 1: Cài đặt thư viện Backend và Frontend**
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

**Bước 2: Khởi chạy FastAPI Backend**
Mở 1 terminal mới và gõ:
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
API nhận diện sẽ chạy tại `http://localhost:8000`. Endpoint để dự đoán là `POST /predict`.

**Bước 3: Khởi chạy Giao diện Frontend Streamlit**
Mở 1 terminal mới (để giữ Backend vẫn chạy ngầm) và gõ:
```bash
streamlit run frontend/streamlit_app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại `http://localhost:8501`. Tại đây bạn có thể kéo thả ảnh thực tế vào để hệ thống nhận diện trái cây và đóng khung Bounding Box.

---

## 🧪 Demo qua Postman (API Testing)

Nếu bạn không muốn dùng giao diện Frontend mà muốn gọi trực tiếp Backend API, hãy sử dụng Postman:

1. Tạo một request chọn phương thức **POST**.
2. URL: `http://localhost:8000/predict`
3. Ở tab **Body**, chọn **form-data**.
4. Ở cột `KEY`, nhập `file` và đổi định dạng từ text sang **File**.
5. Chọn ảnh từ máy tính ở cột `VALUE`.
6. Bấm **Send** và nhận kết quả trả về dạng JSON chứa thông tin tọa độ box và dự đoán Top-5.