import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import time

# ==============================
# Arabic sign names dictionary
# ==============================
arabic_names = {
    0: 'سرعة 20 كم/س',
    1: 'سرعة 30 كم/س',
    2: 'سرعة 50 كم/س',
    3: 'سرعة 60 كم/س',
    4: 'سرعة 70 كم/س',
    5: 'سرعة 80 كم/س', 
    6: 'نهاية سرعة 80',
    7: 'سرعة 100 كم/س',
    8: 'سرعة 120 كم/س',
    9: 'ممنوع التجاوز',
    10: 'ممنوع التجاوز للشاحنات',
    11: 'أولوية عند التقاطع',
    12: 'طريق ذو أولوية',
    13: 'أفسح الطريق',
    14: 'قف',
    15: 'ممنوع المرور',
    16: 'ممنوع للشاحنات',
    17: 'ممنوع الدخول',
    18: 'خطر',
    19: 'منعطف يسار',
    20: 'منعطف يمين',
    21: 'منعطف',
    22: 'طريق وعرة',
    23: 'طريق زلقة',
    24: 'طريق ضيق',
    25: 'أشغال',
    26: 'إشارة ضوئية',
    27: 'عبور مشاة',
    28: 'مدرسة',
    29: 'عبور دراجات',
    30: 'ثلوج',
    31: 'حيوانات',
    32: 'نهاية المنع',
    33: 'اتجه يمين',
    34: 'اتجه يسار',
    35: 'اتجه مباشرة',
    36: 'يمين أو مباشرة',
    37: 'يسار أو مباشرة',
    38: 'الزم اليمين',
    39: 'الزم اليسار',
    40: 'دوار',
    41: 'نهاية منع التجاوز',
    42: 'نهاية منع تجاوز الشاحنات'
}

# ==============================
# Page setup
# ==============================
st.set_page_config(
    page_title="Traffic Sign Detector",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Sign Detection System")
st.markdown("### YOLO Model - 43 Classes (GTSDB)")

# ==============================
# Load model once
# ==============================
@st.cache_resource
def load_model():
    return YOLO("model/traffic_sign_model.pt")

model = load_model()

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
show_details = st.sidebar.checkbox("Show Detection Details", value=True)

# ==============================
# Upload multiple images
# ==============================
uploaded_files = st.file_uploader("📤 Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")

        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        with col1:
            st.image(image, caption="🖼 Original Image", use_column_width=True)

        start_time = time.time()
        with st.spinner("🔍 Detecting traffic signs..."):
            results = model(image_np, conf=confidence)
            annotated = results[0].plot()
        end_time = time.time()
        inference_time = end_time - start_time

        with col2:
            st.image(annotated, caption="🎯 Detection Result", use_column_width=True)

        st.info(f"⏱ Inference Time: {inference_time:.3f} seconds")

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy()
            confidences_vals = boxes.conf.cpu().numpy()
            data = [{"Class": arabic_names[int(cls_id)], "Confidence": round(float(conf_score),2)}
                    for cls_id, conf_score in zip(class_ids, confidences_vals)]
            df = pd.DataFrame(data)
            st.success(f"✅ Total Detections: {len(df)}")
            if show_details:
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠ No traffic signs detected.")