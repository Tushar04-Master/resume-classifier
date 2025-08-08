import streamlit as st
from utils.pdf_extractor import extract_text
import joblib
import os

st.markdown(
    """
    # 📄 AI Resume Classifier  
    Welcome to the **AI-POWERED Resume Classifier**!  
    Upload a resume, and our model will instantly predict the most relevant job role.  

    ---
    **Features**:
    - 📂 Upload resumes in PDF format  
    - 🤖 Machine learning classification  
    - ⚡ Fast and accurate predictions  
    - 📊 Clear results display  

    **Tip:** Use well-formatted resumes for best results.
    """
)

st.title("Resume Classifier")

@st.cache_data
def load_model():
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    mdl = joblib.load("models/xgb_best.pkl")
    lbl = joblib.load("models/label_encoder.pkl")
    return vec, mdl, lbl

def get_sample_resume():
    sample_path = "sample_resumes/sample1.pdf"  # Update path as needed
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            return f.read()
    return None

uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"])
sample_clicked = st.button("Try a Sample Resume")

resume_bytes = None
if uploaded:
    resume_bytes = uploaded
elif sample_clicked:
    sample = get_sample_resume()
    if sample:
        resume_bytes = sample
        st.info("Loaded sample resume!")
    else:
        st.error("Sample resume not found.")

if resume_bytes:
    text = extract_text(resume_bytes)
    st.write("**Extracted Text:**", text[:500], "…")
    vec, mdl, lbl = load_model()
    vec_clean = vec.transform([text])
    pred_proba = mdl.predict_proba(vec_clean)[0]
    pred = mdl.predict(vec_clean)
    decoded = lbl.inverse_transform(pred)[0]
    confidence = max(pred_proba)
    st.success(f"### 🎯 Predicted Category: **{decoded}**")
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
    # Show top 3 classes with scores
    top_indices = pred_proba.argsort()[-3:][::-1]
    st.markdown("#### Top Predictions")
    for idx in top_indices:
        role = lbl.inverse_transform([idx])[0]
        score = pred_proba[idx]
        st.markdown(f"- **{role}**: `{score*100:.2f}%`")