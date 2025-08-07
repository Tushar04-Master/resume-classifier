import streamlit as st
from utils.pdf_extractor import extract_text
import joblib

@st.cache_data
def load_model():
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    mdl = joblib.load("models/xgb_best.pkl")
    lbl = joblib.load("models/label_encoder.pkl")
    return vec, mdl, lbl

st.title("Resume Classifier")
uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"])
if uploaded:
    text = extract_text(uploaded)
    st.write("**Extracted Text:**", text[:500], "…")
    vec, mdl, lbl = load_model()
    vec_clean = vec.transform([text])
    pred = mdl.predict(vec_clean)
    decoded = lbl.inverse_transform(pred)[0]
    st.subheader("Predicted Category:")
    st.write(f"### {decoded}")