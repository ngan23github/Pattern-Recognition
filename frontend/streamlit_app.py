import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io

# --- Config ---
BACKEND_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fruit Recognition App", page_icon="🍎", layout="centered")

st.title("🍎 Fruit Recognition Web App")
st.markdown("Upload an image containing fruits, and our Deep Learning model will recognize them!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # 1. Display uploaded image
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption='Uploaded Image')
    
    if st.button("🔍 Recognize Fruits"):
        with st.spinner("Processing image via FastAPI backend..."):
            try:
                # 2. Send request to backend
                # We need to send the bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                response = requests.post(BACKEND_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        st.warning("No fruits detected.")
                    else:
                        st.success(f"Detected {len(results)} object(s)!")
                        
                        # 3. Draw bounding boxes
                        draw = ImageDraw.Draw(image)
                        colors = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#33FFF0"]
                        
                        st.markdown("### Results Summary")
                        
                        for i, res in enumerate(results):
                            box = res["box"]
                            pred = res["prediction"]
                            color = colors[i % len(colors)]
                            
                            # Draw box
                            draw.rectangle(box, outline=color, width=4)
                            
                            # Draw label
                            label = f"#{res['index']} {pred['top1_class']} ({pred['top1_conf']*100:.1f}%)"
                            
                            # Draw text background
                            text_bbox = draw.textbbox((box[0], max(0, box[1] - 20)), label)
                            draw.rectangle(text_bbox, fill=color)
                            draw.text((box[0], max(0, box[1] - 20)), label, fill="white")
                            
                            # Show summary
                            st.markdown(f"**Object #{res['index']}** (YOLO: {res['detect_label']})")
                            st.write(f"- **Prediction:** {pred['top1_class']} ({pred['top1_conf']*100:.1f}%)")
                            
                            # Show top-k
                            with st.expander("View Top-5 Predictions"):
                                for k in pred['top_k']:
                                    st.write(f" - {k['class']}: {k['confidence']*100:.1f}%")
                            st.divider()

                        # 4. Show image with boxes
                        st.markdown("### Annotated Image")
                        st.image(image, caption="Recognized Fruits")

                else:
                    st.error(f"Error from backend: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}. Is the backend running at {BACKEND_URL}?")
