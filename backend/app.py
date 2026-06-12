import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(
    title="Fruit Recognition API",
    description="API to recognize fruits from images using YOLOv8 and EfficientNet-B0"
)

# --- Configuration ---
BASE_DIR         = Path(__file__).parent
MODEL_PATH       = str(BASE_DIR / '..' / 'models' / 'best_efficientnetB0.pth')
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE         = 100
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
DETECT_CONF      = 0.25
MERGE_IOU_THRESH = 0.05

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# --- Load Models ---
class_names  = []
effnet_model = None
yolo_model   = None

def build_efficientnet_b0(num_classes):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

@app.on_event("startup")
def load_all_models():
    global effnet_model, class_names, yolo_model

    if YOLO_AVAILABLE:
        print("Loading YOLOv8n-seg...")
        yolo_model = YOLO('yolov8n-seg.pt')   # ~7MB, tự download lần đầu
        print("✅ YOLOv8n-seg loaded.")

    print("Loading EfficientNet-B0...")
    if os.path.exists(MODEL_PATH):
        checkpoint   = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names  = checkpoint.get('class_names', [])
        num_classes  = len(class_names)
        effnet_model = build_efficientnet_b0(num_classes).to(DEVICE)
        effnet_model.load_state_dict(checkpoint['model_state_dict'])
        effnet_model.eval()
        print(f"✅ Loaded Fruit Model with {num_classes} classes.")
    else:
        print(f"⚠️  Warning: Model not found at {MODEL_PATH}")

# --- Helper Functions ---
def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Bước 1: Detect + Segmentation (YOLOv8n-seg) ──────────────
def detect_objects(image_pil, conf=DETECT_CONF):
    """
    Dùng YOLOv8n-seg detect object + trả về segmentation mask.
    Fallback: toàn ảnh nếu không detect được gì hoặc YOLO chưa load.
    """
    W, H = image_pil.size

    if yolo_model is None:
        return [np.array([0, 0, W, H])], ['full_image'], [None]

    results    = yolo_model(image_pil, conf=conf, verbose=False)[0]
    boxes_out  = []
    labels_out = []
    masks_out  = []

    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        xyxy   = box.xyxy[0].cpu().numpy().astype(int)
        label  = yolo_model.names[cls_id]
        boxes_out.append(xyxy)
        labels_out.append(label)

        # Trích xuất segmentation mask, scale về kích thước ảnh gốc
        mask = None
        if results.masks is not None and idx < len(results.masks.data):
            mask_tensor = results.masks.data[idx].cpu().numpy()
            mask_pil    = Image.fromarray((mask_tensor * 255).astype(np.uint8))
            mask_pil    = mask_pil.resize((W, H), Image.NEAREST)
            mask        = (np.array(mask_pil) > 127).astype(np.uint8)
        masks_out.append(mask)

    if not boxes_out:
        return [np.array([0, 0, W, H])], ['full_image'], [None]

    return boxes_out, labels_out, masks_out

# ── Bước 2: Merge overlapping same-class boxes ────────────────
def _compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)

def merge_overlapping_boxes(boxes, labels, masks, iou_thresh=MERGE_IOU_THRESH):
    """
    Gom box CÙNG class YOLO có IoU > iou_thresh → box bao trùm + mask union.
    Giải quyết Tình huống 2: 1 dĩa cherry → 1 kết quả duy nhất.
    """
    n      = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j] and _compute_iou(boxes[i], boxes[j]) > iou_thresh:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged_boxes, merged_labels, merged_masks = [], [], []
    for idxs in groups.values():
        x1 = int(min(boxes[i][0] for i in idxs))
        y1 = int(min(boxes[i][1] for i in idxs))
        x2 = int(max(boxes[i][2] for i in idxs))
        y2 = int(max(boxes[i][3] for i in idxs))
        merged_boxes.append(np.array([x1, y1, x2, y2]))
        merged_labels.append(labels[idxs[0]])

        valid = [masks[i] for i in idxs if masks[i] is not None]
        if valid:
            combined = valid[0].copy()
            for m in valid[1:]:
                combined = np.logical_or(combined, m).astype(np.uint8)
            merged_masks.append(combined)
        else:
            merged_masks.append(None)

    return merged_boxes, merged_labels, merged_masks

# ── Bước 3: Crop với mask ─────────────────────────────────────
def crop_with_padding(image_pil, box, mask=None, pad=15):
    """
    Cắt quả bằng segmentation mask nếu có (xóa pixel quả kề — Tình huống 4).
    Fallback về letterbox crop nếu không có mask.
    """
    W, H = image_pil.size
    x1, y1, x2, y2 = [int(v) for v in box]

    if mask is not None:
        # Áp mask: pixel ngoài vùng quả → trắng
        img_arr  = np.array(image_pil).astype(np.uint8)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        bg_arr   = np.full_like(img_arr, 255)
        masked   = np.where(mask_3ch > 0, img_arr, bg_arr)
        x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad);  y2p = min(H, y2 + pad)
        cropped = Image.fromarray(masked).crop((x1p, y1p, x2p, y2p))
    else:
        # Letterbox crop (hành vi cũ)
        side = max(x2 - x1, y2 - y1) + pad * 2
        cx   = (x1 + x2) // 2;  cy = (y1 + y2) // 2
        x1p  = max(0, cx - side // 2);  y1p = max(0, cy - side // 2)
        x2p  = min(W, cx + side // 2);  y2p = min(H, cy + side // 2)
        cropped = image_pil.crop((x1p, y1p, x2p, y2p))

    # Đặt vào khung vuông (không méo khi resize về 100×100)
    w_c, h_c  = cropped.size
    side       = max(w_c, h_c)
    square_img = Image.new('RGB', (side, side), (255, 255, 255))
    square_img.paste(cropped, ((side - w_c) // 2, (side - h_c) // 2))
    return square_img

# ── Bước 3b: Pre-processing ───────────────────────────────────
def preprocess_crop(image_pil):
    """
    White balance (per-channel histogram stretch) + contrast ×1.2.
    Giảm domain gap do ánh sáng thực tế (Tình huống 1 & 3).
    """
    img = np.array(image_pil).astype(np.float32)
    for c in range(3):
        ch     = img[:, :, c]
        non_bg = ch[ch < 245]
        if len(non_bg) > 200:
            lo, hi = np.percentile(non_bg, 2), np.percentile(non_bg, 98)
            if hi > lo + 5:
                img[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
    result = Image.fromarray(img.astype(np.uint8))
    result = ImageEnhance.Contrast(result).enhance(1.2)
    return result

# ── Bước 3c: Remove background (fallback khi không có mask) ───
def remove_background(image_pil, has_mask=False):
    """Dùng rembg chỉ khi không có YOLO mask."""
    if has_mask or not REMBG_AVAILABLE:
        return image_pil.convert('RGB')
    try:
        img_rgba = rembg_remove(
            image_pil,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        background = Image.new('RGB', img_rgba.size, (255, 255, 255))
        background.paste(img_rgba, mask=img_rgba.split()[3])
        return background
    except Exception as e:
        print(f"rembg error: {e}")
        return image_pil.convert('RGB')

@torch.no_grad()
def classify_crop(crop_pil, top_k=5):
    if effnet_model is None:
        return {"error": "Model not loaded"}

    tensor     = val_test_transforms(crop_pil).unsqueeze(0).to(DEVICE)
    logits     = effnet_model(tensor)
    probs      = torch.softmax(logits, dim=1)
    top_probs, top_idxs = probs.topk(top_k, dim=1)
    top_probs  = top_probs.squeeze().cpu().numpy()
    top_idxs   = top_idxs.squeeze().cpu().numpy()

    results = [
        {'class': class_names[i], 'confidence': float(p)}
        for p, i in zip(top_probs, top_idxs)
    ]
    return {
        'top1_class': results[0]['class'],
        'top1_conf' : results[0]['confidence'],
        'top_k'     : results
    }

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Fruit Recognition API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if effnet_model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Fruit model is not loaded (check models/best_efficientnetB0.pth)."}
        )

    try:
        image_bytes = await file.read()
        image_pil   = Image.open(io.BytesIO(image_bytes))
        image_pil   = ImageOps.exif_transpose(image_pil).convert('RGB')
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image file: {str(e)}"})

    # Bước 1: Detect + Segmentation
    boxes, detect_labels, masks = detect_objects(image_pil)

    # Bước 2: Merge overlapping same-class boxes
    boxes, detect_labels, masks = merge_overlapping_boxes(boxes, detect_labels, masks)

    all_results = []
    for i, (box, det_label, mask) in enumerate(zip(boxes, detect_labels, masks)):
        has_mask = mask is not None

        # Bước 3: Crop với mask
        crop = crop_with_padding(image_pil, box, mask=mask)

        # Bước 3b: Pre-processing
        crop = preprocess_crop(crop)

        # Bước 3c: Remove background (fallback nếu không có mask)
        crop_clean = remove_background(crop, has_mask=has_mask)

        # Bước 4: Classify
        result = classify_crop(crop_clean)

        # Encode crop về base64 (giữ nguyên như cũ)
        crop_b64 = pil_to_base64(crop_clean, fmt="PNG")

        all_results.append({
            'index'        : i + 1,
            'box'          : box.tolist(),
            'detect_label' : det_label,
            'has_seg_mask' : has_mask,
            'prediction'   : result,
            'crop_b64'     : crop_b64,
        })

    return {"n_objects": len(all_results), "results": all_results}