import os
import traceback

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------- Page setup ----------
st.set_page_config(page_title="Mammography Portal", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = [
    "breast_mammography_model_rebuilt.keras",
    "breast_mammography_model.keras",
    "breast_mammography_model.h5",
]
IMG_SIZE = (150, 150)

# ---------- Custom style ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .hero {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p  { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.05rem; }
    .notice {
        background: #fff7e6;
        border-left: 6px solid #f59e0b;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        color: #7a4b00;
        font-size: 0.93rem;
        margin-bottom: 1.2rem;
    }
    .result-card {
        padding: 1.3rem 1.4rem;
        border-radius: 16px;
        margin-bottom: 0.8rem;
    }
    .malignant { background: #fef2f2; border: 2px solid #ef4444; }
    .benign    { background: #f0fdf4; border: 2px solid #22c55e; }
    .result-card h2 { margin: 0 0 0.2rem 0; font-size: 1.6rem; }
    .malignant h2 { color: #b91c1c; }
    .benign h2    { color: #15803d; }
    .result-card p { margin: 0; color: #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>Mammography Diagnostic Portal</h1>
        <p>Breast Cancer Lesion Verification Subsystem</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <b>MANDATORY ACADEMIC NOTICE:</b> This application is a coursework prototype
        for an AkiraChix DAS assignment. It is NOT an FDA-approved medical tool and has
        not been clinically validated. It is completely unsafe for use on real patients.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Technical Profile and Target Modality", expanded=False):
    st.markdown(
        """
        - **Target Modality:** Screening Mammography (X-ray scans)
        - **Problem Profile:** Binary image classification (Benign vs. Malignant)
        - **Input Size:** 150 x 150 pixels, RGB
        """
    )


# ---------- Model loading ----------
@st.cache_resource(show_spinner="Loading model...")
def load_one_model(path):
    """Load one model file. Raise a clear error if it fails."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        raise RuntimeError(traceback.format_exc())


def get_model():
    """Return (model, name, errors, found_any)."""
    errors = []
    found_any = False
    for name in MODEL_FILES:
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            continue
        found_any = True
        try:
            m = load_one_model(path)
            return m, name, errors, found_any
        except Exception as e:
            errors.append(f"File: {name}\n\n{e}")
    return None, None, errors, found_any


model, model_name, load_errors, found_any = get_model()

with st.sidebar:
    st.header("System Status")
    if model is not None:
        st.success(f"Model loaded: {model_name}")
    else:
        st.error("Model not loaded")
    st.divider()
    st.caption(f"TensorFlow: {tf.__version__}")
    st.caption(f"Folder: {BASE_DIR}")

if model is None:
    if not found_any:
        st.error(
            "No model file was found. Put `breast_mammography_model_rebuilt.keras` "
            f"in this folder: {BASE_DIR}"
        )
    else:
        st.error("A model file was found, but it could not be loaded. Full error below:")
        for err in load_errors:
            st.code(err if err.strip() else "(empty error message)", language="text")


# ---------- Helper functions ----------
def prepare_image(pil_img):
    """Resize, convert to float, apply ResNet50 preprocessing, add batch dimension."""
    rgb = pil_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(rgb, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def get_malignant_score(model, batch):
    """Handle both 1-unit sigmoid and 2-unit softmax outputs."""
    pred = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if pred.size == 1:
        return float(pred[0])
    return float(pred[1])


# ---------- Upload ----------
st.markdown("### Upload a Scan")
uploaded_file = st.file_uploader(
    "Choose a screening mammography image (PNG or JPG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
    except Exception:
        st.error("Could not read this image. Please upload a valid PNG or JPG file.")
        st.stop()

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.image(pil_img, caption="Uploaded Mammogram Preview", use_container_width=True)

    with col2:
        if model is None:
            st.info("Analysis is not available until the model loads correctly.")
        else:
            try:
                with st.spinner("Running evaluation pipeline..."):
                    batch = prepare_image(pil_img)
                    score = get_malignant_score(model, batch)

                st.markdown("### Analysis Results")

                if score >= 0.5:
                    label, confidence, css = "Malignant", score * 100, "malignant"
                else:
                    label, confidence, css = "Benign", (1.0 - score) * 100, "benign"

                st.markdown(
                    f"""
                    <div class="result-card {css}">
                        <h2>{label}</h2>
                        <p>Diagnostic Category</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric("Model Confidence Score", f"{confidence:.2f}%")
                st.progress(min(max(confidence / 100, 0.0), 1.0))

                st.write(
                    f"The model is **{confidence:.1f}%** confident that the patterns in "
                    f"this mammogram match a **{label.lower()}** classification profile."
                )
                st.caption(f"Raw model output (malignant probability): {score:.4f}")

            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.code(traceback.format_exc(), language="text")
else:
    st.info("Upload an image above to start the analysis.")

st.divider()
st.caption("Academic prototype only. Not for clinical use.")