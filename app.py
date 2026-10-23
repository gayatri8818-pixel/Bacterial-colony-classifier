
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

BASE = Path(__file__).parent
MODEL_PATH = BASE / "bacterial_colony_classifier.keras"
CLASSES_PATH = BASE / "colony_class_names.json"

st.set_page_config(
    page_title="Bacterial Colony Classifier",
    page_icon="🧫"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        if all(str(k).isdigit() for k in labels):
            labels = [
                labels[k] if k in labels else labels[str(k)]
                for k in sorted(labels, key=lambda x: int(x))
            ]
        elif all(isinstance(v, (int, float)) for v in labels.values()):
            labels = [
                k for k, v in sorted(
                    labels.items(), key=lambda item: item[1]
                )
            ]
        else:
            labels = list(labels.values())

    return labels

st.title("🧫 Bacterial Colony Classification")
st.write(
    "Upload a bacterial colony image to classify it "
    "using a trained deep learning model."
)

uploaded = st.file_uploader(
    "Upload a colony image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Classify image"):
        with st.spinner("Classifying..."):
            model = load_model()
            class_names = load_classes()

            image = image.resize((128, 128))
            array = np.asarray(image, dtype=np.float32)[None, ...]
            predictions = model.predict(array, verbose=0)[0]
            top = np.argsort(predictions)[-5:][::-1]
            best = int(top[0])

        st.subheader("Prediction")
        st.write(f"**Class:** {class_names[best]}")
        st.write(f"**Confidence:** {predictions[best]:.2%}")

        st.subheader("Top 5 predictions")
        for i in top:
            st.write(
                f"{class_names[i]} — {predictions[i]:.2%}"
            )
