import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO("yolov10n.pt")

# Class definitions
biodegradable_items = [
    'banana', 'apple', 'broccoli', 'carrot', 'sandwich',
    'orange', 'book', 'pizza', 'donut', 'cake', 'vegetable',
    'fruit', 'hot dog', 'bread', 'meat', 'fish', 'egg'
]

non_biodegradable_items = [
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def classify_items(labels):
    bio = []
    non_bio = []
    for label in labels:
        label = label.lower()
        if any(b in label for b in biodegradable_items):
            bio.append(label)
        elif any(nb in label for nb in non_biodegradable_items):
            non_bio.append(label)
    return bio, non_bio

# Streamlit UI
st.title("♻️ Smart Waste Classifier")
st.write("Upload an image to classify detected objects as biodegradable or non-biodegradable.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL and convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO model
    results = model(image_bgr)

    st.markdown("### 🎯 Detected Objects & Classification:")
    for r in results:
        labels = [r.names[int(cls)] for cls in r.boxes.cls]
        bio, non_bio = classify_items(labels)
        for i, cls in enumerate(labels):
            confidence = r.boxes.conf[i].item()
            if cls.lower() in [b.lower() for b in bio]:
                st.markdown(f"✅ **{cls}** ({confidence:.2f}) → Biodegradable")
            elif cls.lower() in [nb.lower() for nb in non_bio]:
                st.markdown(f"❌ **{cls}** ({confidence:.2f}) → Non-Biodegradable")
            else:
                st.markdown(f"❓ **{cls}** ({confidence:.2f}) → Unknown")

    # Optionally display bounding boxes (drawn using OpenCV)
    for r in results:
        annotated_frame = r.plot()
        st.image(annotated_frame, caption="YOLOv10 Detections", use_column_width=True)
