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
    Proyek ini bertujuan untuk mengklasifikasikan teks terkait kesehatan mental
    (misalnya cuitan di Twitter/X) ke dalam beberapa kategori, seperti:

    - Anxiety  
    - Depression  
    - Stress  
    - Bipolar  
    - Suicidal  
    - Normal  

    **Model yang digunakan:**
    - 🧮 *Model klasik*: TF-IDF + Logistic Regression  
    - 🤖 *Model Transformers*: Model bahasa pretrained yang di-finetune  

    Proyek ini dibuat untuk:
    - Mengeksplorasi perbandingan model klasik vs Transformers  
    - Mendemonstrasikan end-to-end pipeline NLP (preprocessing → training → deployment dengan Streamlit)
    """
)

st.title("MindPulse-X 🧠")
st.markdown("---")
st.markdown("## Mental Health Text Classification")

model_choice = st.radio(
    "Pilih model yang ingin digunakan:",
    ("Model klasik (TF-IDF + Logistic Regression)", "Model Transformers"),
)

user_text = st.text_area("Masukkan teks ✏️:", height=150)

if st.button("Klasifikasikan") and user_text.strip():
    if model_choice.startswith("Model klasik"):
        pred_label, probs, labels = predict_classic(user_text)
        st.subheader(f"Prediksi utama (Model Klasik): **{pred_label}**")

        if probs is not None and labels is not None:
            st.write("Probabilitas tiap label (Model Klasik):")
            for lbl, p in zip(labels, probs):
                st.write(f"- **{lbl}**: `{p:.3f}`")

    else:
        pred_label, probs, labels_map = predict_transformer(user_text)
        st.subheader(f"Prediksi utama (Transformers): **{pred_label}**")

        st.write("Probabilitas tiap label (Transformers):")
        # labels_map = dict {0: 'Anxiety', 1: 'Bipolar', ...}
        for i, p in enumerate(probs):
            st.write(f"- **{labels_map[i]}**: `{p:.3f}`")
