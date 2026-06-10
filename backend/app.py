import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(title="Fruit Recognition API", description="API to recognize fruits from images using YOLOv8 and EfficientNet")

# --- Configuration ---
BASE_DIR   = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / '..' / 'models' / 'best_efficientnetB0.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 100
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# --- Load Models ---
class_names = []
effnet_model = None
yolo_model = None

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
        print("Loading YOLOv8n...")
        yolo_model = YOLO('yolov8n.pt')
    
    print("Loading EfficientNet-B0...")
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint.get('class_names', [])
        num_classes = len(class_names)
        effnet_model = build_efficientnet_b0(num_classes).to(DEVICE)
        effnet_model.load_state_dict(checkpoint['model_state_dict'])
        effnet_model.eval()
        print(f"✅ Loaded Fruit Model with {num_classes} classes.")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")

# --- Helper Functions ---
def detect_objects(image_pil, conf=0.25):
    W, H = image_pil.size
    if yolo_model is None:
        return [np.array([0, 0, W, H])], ['full_image']
    
    results = yolo_model(image_pil, conf=conf, verbose=False)[0]
    boxes_out, labels_out = [], []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        xyxy   = box.xyxy[0].cpu().numpy().astype(int)
        label  = yolo_model.names[cls_id]
        boxes_out.append(xyxy)
        labels_out.append(label)
        
    if not boxes_out:
        boxes_out = [np.array([0, 0, W, H])]
        labels_out = ['full_image']
    return boxes_out, labels_out

def crop_with_padding(image_pil, box, pad=15):
    W, H = image_pil.size
    x1, y1, x2, y2 = box
    w_box = x2 - x1
    h_box = y2 - y1
    side = max(w_box, h_box) + pad * 2
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    x1p = max(0, center_x - side // 2)
    y1p = max(0, center_y - side // 2)
    x2p = min(W, center_x + side // 2)
    y2p = min(H, center_y + side // 2)
    
    cropped = image_pil.crop((x1p, y1p, x2p, y2p))
    square_img = Image.new('RGB', (side, side), (255, 255, 255))
    paste_x = (side - cropped.size[0]) // 2
    paste_y = (side - cropped.size[1]) // 2
    square_img.paste(cropped, (paste_x, paste_y))
    return square_img

def remove_background(image_pil):
    if not REMBG_AVAILABLE:
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
    
    tensor = val_test_transforms(crop_pil).unsqueeze(0).to(DEVICE)
    logits = effnet_model(tensor)
    probs  = torch.softmax(logits, dim=1)
    
    top_probs, top_idxs = probs.topk(top_k, dim=1)
    top_probs = top_probs.squeeze().cpu().numpy()
    top_idxs  = top_idxs.squeeze().cpu().numpy()
    
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
        return JSONResponse(status_code=500, content={"error": "Fruit model is not loaded (check models/best_efficientnetB0.pth)."})
        
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_pil = ImageOps.exif_transpose(image_pil).convert('RGB')
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image file: {str(e)}"})

    # 1. Detect
    boxes, detect_labels = detect_objects(image_pil)
    
    all_results = []
    for i, (box, det_label) in enumerate(zip(boxes, detect_labels)):
        # 2. Crop
        crop = crop_with_padding(image_pil, box)
        
        # 3. Remove Background
        crop_clean = remove_background(crop)
        
        # 4. Classify
        result = classify_crop(crop_clean)
        
        all_results.append({
            'index': i + 1,
            'box': box.tolist(),
            'detect_label': det_label,
            'prediction': result
        })

    return {"results": all_results}
