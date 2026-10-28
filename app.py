"""Plant Disease Detection System – Streamlit web app."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path("model/plant_disease_model.keras")
CLASSES_PATH = Path("model/class_names.json")
IMG_SIZE = 224

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------- knowledge base
DISEASE_INFO = [
    ("apple scab", "Fungal disease causing olive-green to brown scabby spots on leaves and fruit.",
     "Remove fallen leaves, prune for airflow, apply fungicide (captan/sulfur) from bud break."),
    ("cedar apple rust", "Fungal disease producing bright yellow-orange spots on leaves.",
     "Remove nearby juniper/cedar hosts, use resistant varieties, apply fungicide in spring."),
    ("black rot", "Fungal disease causing dark circular lesions on leaves and rotting fruit.",
     "Prune infected wood, destroy mummified fruit, apply protectant fungicide."),
    ("esca", "Complex fungal trunk disease causing 'tiger-stripe' leaves and dark spotted fruit.",
     "Prune in dry weather, disinfect tools, remove severely infected vines."),
    ("isariopsis", "Fungal leaf spot causing dark angular lesions and early defoliation.",
     "Improve canopy airflow, remove infected leaves, apply copper-based fungicide."),
    ("haunglongbing", "Citrus greening: bacterial disease spread by psyllids; causes mottled leaves and bitter fruit.",
     "No cure. Remove infected trees, control psyllid insects, plant certified disease-free stock."),
    ("powdery mildew", "Fungal disease forming white powdery coating on leaves.",
     "Improve air circulation, avoid overhead watering, apply sulfur or potassium bicarbonate spray."),
    ("gray leaf spot", "Fungal disease producing long rectangular gray-tan lesions on corn leaves.",
     "Rotate crops, till residue, plant resistant hybrids, fungicide if severe."),
    ("common rust", "Fungal disease with small reddish-brown pustules on both leaf surfaces.",
     "Plant resistant hybrids; apply fungicide early if infection is heavy."),
    ("northern leaf blight", "Fungal disease causing long cigar-shaped gray-green lesions.",
     "Use resistant hybrids, rotate crops, bury crop residue, apply fungicide if needed."),
    ("leaf scorch", "Fungal disease causing purple-brown spots that merge so leaves look scorched.",
     "Remove infected leaves, avoid overhead irrigation, use resistant varieties."),
    ("bacterial spot", "Bacterial disease causing small dark water-soaked spots on leaves and fruit.",
     "Use disease-free seed, copper-based sprays, avoid working with wet plants, rotate crops."),
    ("early blight", "Fungal disease with brown spots showing concentric 'target' rings.",
     "Remove lower infected leaves, mulch, rotate crops, apply chlorothalonil or copper fungicide."),
    ("late blight", "Water-mold disease causing large dark water-soaked lesions; spreads very fast in cool, wet weather.",
     "Remove and destroy infected plants, avoid wet foliage, apply preventive fungicide, use resistant varieties."),
    ("leaf mold", "Fungal disease: yellow spots on top of leaf, olive mold underneath (humid conditions).",
     "Reduce humidity, ventilate greenhouses, space plants, apply fungicide."),
    ("septoria", "Fungal disease with many small circular spots with dark borders and gray centres.",
     "Remove infected lower leaves, mulch, rotate crops, apply fungicide."),
    ("spider mite", "Tiny mites causing stippling, yellowing and fine webbing on leaves.",
     "Spray leaves with water, use insecticidal soap/neem oil, encourage predatory mites."),
    ("target spot", "Fungal disease causing brown lesions with concentric rings on leaves and fruit.",
     "Improve airflow, avoid overhead watering, remove debris, apply fungicide."),
    ("yellow leaf curl", "Virus spread by whiteflies: upward leaf curling, yellowing and stunted growth.",
     "Control whiteflies, use reflective mulch/nets, remove infected plants, plant resistant varieties."),
    ("mosaic", "Viral disease causing mottled light/dark green leaves and distortion.",
     "No cure. Remove infected plants, disinfect tools, wash hands, use virus-free seed."),
]


def prettify(class_name: str):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    crop, _, disease = class_name.partition("___")
    clean = lambda s: re.sub(r"\s+", " ", s.replace("_", " ").replace(",", "")).strip()
    return clean(crop), clean(disease) if disease else ""


def lookup_info(disease: str):
    d = disease.lower()
    if "healthy" in d:
        return ("The leaf looks healthy.",
                "No treatment needed. Keep up good watering, nutrition and regular monitoring.")
    for key, desc, tip in DISEASE_INFO:
        if key in d:
            return desc, tip
    return "No description available for this class.", "Consult a local agriculture officer."


def is_likely_plant_leaf(img: Image.Image) -> bool:
    """Strictly checks if the image is a plant leaf based on natural foliage color distribution."""
    img_rgb = img.convert("RGB").resize((100, 100))
    arr = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Filter out pure white or light grey backgrounds
    is_not_white = (r < 240) | (g < 240) | (b < 240)
    if np.sum(is_not_white) < 100:
        return False
        
    # Plant leaves have green significantly higher than red and blue (chlorophyll profile)
    # Using a stricter margin (g > r + 5 and g > b + 5) to block teal/cyan car lighting
    green_dominant = (g > r + 5) & (g > b + 5)
    green_ratio = np.sum(green_dominant & is_not_white) / np.sum(is_not_white)
    
    # Require at least 25% strong green-dominant pixels for a valid leaf
    return green_ratio > 0.25


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_classes():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text())
    return model, classes


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def predict(model, classes, img):
    probs = model.predict(preprocess(img), verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(classes[i], float(probs[i])) for i in top]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🌾 About")
    st.write(
        "Upload or capture a photo of a plant leaf and the model will identify the "
        "crop and any disease."
    )
    st.markdown(
        "- **Model:** MobileNetV2 / EfficientNetB0 (transfer learning)\n"
        "- **Dataset:** PlantVillage (38 classes)\n"
        "- **Deployment:** Streamlit\n"
        "- **SDG 2:** Zero Hunger – early detection protects crop yield"
    )
    st.header("📸 Tips")
    st.write("Use a single, well-lit leaf close to the camera on a plain background.")

# ---------------------------------------------------------------- main page
st.title("🌿 Plant Disease Detection System")
st.caption("AI-powered leaf disease diagnosis for healthier crops and food security.")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Trained model not found.")
    st.markdown(
        "Run training first:\n```bash\npython train.py --data_dir path/to/PlantVillage/color --model mobilenetv2\n```\n"
        "This creates `model/plant_disease_model.keras` and `model/class_names.json`."
    )
    st.stop()

model, classes = load_model_and_classes()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("1. Provide a leaf image")
    source = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)
    file = (
        st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"])
        if source == "Upload image"
        else st.camera_input("Take a picture")
    )
    image = Image.open(file) if file else None
    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

with right:
    st.subheader("2. Diagnosis")
    if image is None:
        st.info("Waiting for an image…")
    else:
        # Check if the uploaded image has plant-like characteristics first
        if not is_likely_plant_leaf(image):
            st.error("⚠️ **Invalid Image / Non-Leaf Detected!**")
            st.warning(
                "The uploaded image does not appear to be a plant leaf (it lacks natural plant color profiles). "
                "Please upload a clear photo of a plant leaf."
            )
        else:
            with st.spinner("Analysing leaf…"):
                results = predict(model, classes, image)
            best_name, best_conf = results[0]
            crop, disease = prettify(best_name)
            is_healthy = "healthy" in disease.lower()

            m1, m2 = st.columns(2)
            m1.metric("Crop", crop)
            m2.metric("Confidence", f"{best_conf * 100:.1f}%")

            if is_healthy:
                st.success("✅ Healthy plant")
            else:
                st.error(f"🦠 Detected: {disease}")

            desc, tip = lookup_info(disease)
            st.markdown(f"**About:** {desc}")
            st.markdown(f"**Recommended action:** {tip}")

            st.markdown("**Top 3 predictions**")
            df = pd.DataFrame(
                {"Class": [" – ".join(filter(None, prettify(n))) for n, _ in results],
                 "Probability (%)": [round(p * 100, 2) for _, p in results]}
            ).set_index("Class")
            st.bar_chart(df)

st.divider()
st.caption(
    "⚠️ This tool is a decision-support aid trained on lab-style images. "
    "For serious outbreaks consult an agricultural expert."
)