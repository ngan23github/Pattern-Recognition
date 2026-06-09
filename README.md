# 🍎 Nhận Diện Trái Cây — Pattern Recognition

Đồ án môn **Nhận Dạng Mẫu**, triển khai và so sánh các mô hình Deep Learning cho bài toán phân loại trái cây trên dataset **Fruit 360**, kết hợp pipeline nhận diện ảnh thực tế.

---

## 📌 Tổng Quan

| Hạng mục | Chi tiết |
|---|---|
| Dataset | [Fruit 360](https://www.kaggle.com/datasets/moltean/fruits) |
| Framework | PyTorch |
| Mô hình so sánh | Custom CNN · ResNet-18 · EfficientNet-B0 |
| Inference thực tế | YOLOv8 (detect) + rembg (remove background) + Classifier |

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
| Fruit 360 | data gốc, ~90k ảnh, nền trắng | [Kaggle](https://www.kaggle.com/datasets/moltean/fruits) |
| Data đang dùng | Data đã được merged lại với nhau | [Google Drive](https://drive.google.com/file/d/1_FWHCbA7YflZKAl0ogdZBJErEHZ8G58T/view?usp=drive_link) |

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

## ⚙️ Cài Đặt

```bash
# Clone repo
git clone https://github.com/ngan23github/Pattern-Recognition.git
cd Pattern-Recognition

# Tạo môi trường ảo
python -m venv venv
source venv/Scripts/activate   # Git Bash (Windows)

# Cài thư viện
pip install torch torchvision tqdm scikit-learn matplotlib pillow seaborn rembg ultralytics
```

---

## 🚀 Hướng Dẫn Chạy

### Huấn luyện & đánh giá
Mở và chạy tuần tự `fruit_recognition_3models.ipynb`.  
Notebook tự động áp dụng **Early Stopping** (patience=7) cho cả 3 model — dừng training khi val loss không cải thiện sau 7 epoch liên tiếp.

---

## 📊 Kết Quả

> Cập nhật sau khi hoàn thành training.

| Model | Test Top-1 Acc | Test Top-5 Acc | Epochs thực chạy |
|---|---|---|---|
| Custom CNN | — | — | — |
| ResNet-18 | — | — | — |
| EfficientNet-B0 | — | — | — |