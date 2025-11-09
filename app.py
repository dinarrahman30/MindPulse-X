import json
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# KONFIGURASI
# ======================
TRANSFORMER_DIR = "models"
CLASSIC_PATH = "saved_models/logistic_regression_pipeline.joblib"  # sesuaikan kalau beda lokasi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL TRANSFORMERS
# ======================
@st.cache_resource
def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    model.to(device)
    model.eval()

    with open(f"{TRANSFORMER_DIR}/label_mapping.json") as f:
        mapping = json.load(f)

    id2label_raw = mapping["id2label"]
    id2label = {int(k): v for k, v in id2label_raw.items()}
    return tokenizer, model, id2label

# ======================
# LOAD MODEL KLASIK (PIPELINE)
# ======================
@st.cache_resource
def load_classic_model():
    pipeline = joblib.load(CLASSIC_PATH)
    return pipeline

tokenizer_tf, model_tf, id2label_tf = load_transformer_model()
classic_pipeline = load_classic_model()

# ======================
# FUNGSI PREDIKSI
# ======================
def predict_transformer(text: str):
    inputs = tokenizer_tf(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model_tf(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        pred_id = torch.argmax(logits, dim=-1).item()
        pred_label = id2label_tf[pred_id]

    return pred_label, probs, id2label_tf  # dict: id -> label

def predict_classic(text: str):
    X = [text]
    pred = classic_pipeline.predict(X)[0]

    probs = None
    labels = None
    if hasattr(classic_pipeline, "predict_proba"):
        probs = classic_pipeline.predict_proba(X)[0].tolist()
        labels = classic_pipeline.classes_.tolist()  # label string: ['Anxiety', 'Bipolar', ...]

    return pred, probs, labels

# ======================
# STREAMLIT UI
# ======================
st.sidebar.title("ℹ️ Tentang Proyek Ini")
st.sidebar.markdown(
    """
    **Mental Health Sentiment Analysis**
    This project aims to classify text related to mental health
    (for example, tweets from Twitter/X) into several categories, such as:

    - Anxiety
    - Depression
    - Stress
    - Bipolar
    - Suicidal
    - Normal

    Models used:
    - 🧮 Classical model: TF-IDF + Logistic Regression
    - 🤖 Transformers model: Fine-tuned pretrained language model

    This project is built to:
    - Explore a comparison between classical models and Transformers
    - Demonstrate an end-to-end NLP pipeline (preprocessing → training → deployment with Streamlit)
    """
)

st.title("MindPulse-X 🧠")
st.markdown("---")
st.markdown("## Mental Health Text Classification")

model_choice = st.radio(
    "Select the model you want to use:",
    ("Classic model (TF-IDF + Logistic Regression)", "Transformers Model"),
)

user_text = st.text_area("Input text ✏️:", height=150)

if st.button("Classify") and user_text.strip():
    if model_choice.startswith("Classic Model"):
        pred_label, probs, labels = predict_classic(user_text)
        st.subheader(f"Main prediction (Classic Model): **{pred_label}**")

        if probs is not None and labels is not None:
            st.write("Probability of each label (Classic Model):")
            for lbl, p in zip(labels, probs):
                st.write(f"- **{lbl}**: `{p:.3f}`")

    else:
        pred_label, probs, labels_map = predict_transformer(user_text)
        st.subheader(f"Main prediction (Transformers): **{pred_label}**")

        st.write("Probability of each label (Transformers):")
        # labels_map = dict {0: 'Anxiety', 1: 'Bipolar', ...}
        for i, p in enumerate(probs):
            st.write(f"- **{labels_map[i]}**: `{p:.3f}`")
