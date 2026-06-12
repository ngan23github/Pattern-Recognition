import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageOps, ImageFont
import io
import base64

# --- Config ---
BACKEND_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fruit Recognition App", page_icon="🍎", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1200px; padding: 2rem 3rem; }

/* Fixed-height image preview box */
.preview-box-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 0.75rem;
}
.preview-box {
    max-height: 320px;
    width: fit-content;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 1px solid #ddd;
    border-radius: 10px;
    background: #fafafa;
}
.preview-box img {
    max-height: 320px;
    max-width: 100%;
    object-fit: contain;
    border-radius: 8px;
}

/* Recognize button style */
div[data-testid="stButton"] > button {
    font-size: 1.5rem;
    font-weight: bold;
    padding: 0.65rem 1.5rem;
    border-radius: 10px;
    background-color: #A8D8EA !important;
    color: #1a3a4a !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(168,216,234,0.5) !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #85C7DF !important;
    box-shadow: 0 4px 12px rgba(168,216,234,0.7) !important;
}

/* Badge styles */
.badge-seg {
    display: inline-block;
    background: #d4edda; color: #1a5c2e;
    font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 12px;
    vertical-align: middle; margin-left: 4px;
}
.badge-bbox {
    display: inline-block;
    background: #fff3cd; color: #856404;
    font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 12px;
    vertical-align: middle; margin-left: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🍎 Fruit Recognition Web App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload an image containing fruits, and our Deep Learning model will recognize them!</p>", unsafe_allow_html=True)

_, upload_col, _ = st.columns([1, 2, 1])
with upload_col:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f'<div class="preview-box-wrap"><div class="preview-box">'
            f'<img src="data:image/png;base64,{b64}" alt="Uploaded image">'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        _, btn_mid, _ = st.columns([1, 1, 1])
        with btn_mid:
            st.button("🔍 Recognize Fruits", key="recognize_btn")

if uploaded_file is not None and st.session_state.get("recognize_btn"):
    with st.spinner("Processing image via FastAPI backend..."):
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()

            files    = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
            response = requests.post(BACKEND_URL, files=files)

            if response.status_code == 200:
                data      = response.json()
                results   = data.get("results", [])
                n_objects = data.get("n_objects", len(results))

                if not results:
                    st.warning("No fruits detected.")
                else:
                    st.success(f"Detected **{n_objects}** object(s)!")

                    # ── Draw bounding boxes on annotated copy ──────────
                    annotated = image.copy()
                    draw      = ImageDraw.Draw(annotated)
                    colors    = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#33FFF0"]

                    font_size = max(20, annotated.width // 30)
                    try:
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
                        )
                    except Exception:
                        font = ImageFont.load_default(size=font_size)

                    for i, res in enumerate(results):
                        box      = res["box"]
                        pred     = res["prediction"]
                        color    = colors[i % len(colors)]
                        has_mask = res.get("has_seg_mask", False)
                        mode_tag = "[seg]" if has_mask else "[bbox]"
                        label    = f"#{res['index']} {pred['top1_class']} ({pred['top1_conf']*100:.1f}%) {mode_tag}"

                        draw.rectangle(box, outline=color, width=max(3, font_size // 8))
                        text_bbox = draw.textbbox(
                            (box[0], max(0, box[1] - font_size - 4)), label, font=font
                        )
                        draw.rectangle(
                            [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                            fill=color,
                        )
                        draw.text(
                            (box[0], max(0, box[1] - font_size - 4)),
                            label, fill="white", font=font
                        )

                    # ── Two-column image view ──────────────────────────
                    st.markdown("## Results")
                    col_orig, col_ann = st.columns(2)
                    with col_orig:
                        st.caption("Original")
                        st.image(image)
                    with col_ann:
                        st.caption("Detected regions")
                        st.image(annotated)

                    # ── Per-object detail cards — 2 per row ───────────
                    st.markdown("### 🔍 Predictions by Region")
                    for row_start in range(0, len(results), 2):
                        card_left, card_right = st.columns(2, gap="medium")
                        for col, res in zip(
                            [card_left, card_right],
                            results[row_start : row_start + 2],
                        ):
                            pred     = res["prediction"]
                            color    = colors[(res["index"] - 1) % len(colors)]
                            has_mask = res.get("has_seg_mask", False)

                            # Badge HTML
                            if has_mask:
                                badge = '<span class="badge-seg">✅ Segmentation</span>'
                            else:
                                badge = '<span class="badge-bbox">⚠️ BBox fallback</span>'

                            with col:
                                with st.expander(
                                    f"#{res['index']} — {pred['top1_class']}"
                                    f"  ({pred['top1_conf']*100:.1f}%)"
                                    f"  ·  YOLO: {res['detect_label']}",
                                    expanded=True,
                                ):
                                    # Badge + mode line
                                    st.markdown(
                                        f"Isolation mode: {badge}",
                                        unsafe_allow_html=True,
                                    )

                                    thumb_col, info_col = st.columns([1, 2])

                                    with thumb_col:
                                        crop_b64 = res.get("crop_b64", "")
                                        if crop_b64:
                                            crop_bytes = base64.b64decode(crop_b64)
                                            crop_img   = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
                                            st.image(crop_img, caption="Cropped region")
                                        else:
                                            st.info("No crop available")

                                    with info_col:
                                        st.markdown("**Top-5 Predictions**")
                                        for k in pred["top_k"]:
                                            pct = k["confidence"] * 100
                                            st.progress(
                                                min(int(pct), 100),
                                                text=f"{k['class']}  —  {pct:.1f}%",
                                            )

            else:
                st.error(f"Error from backend: {response.text}")

        except Exception as e:
            st.error(f"Connection error: {e}. Is the backend running at {BACKEND_URL}?")