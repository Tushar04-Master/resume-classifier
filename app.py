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

# Sidebar About section
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.write("""
    This app predicts the category of a resume using Machine Learning.
             
    Built with Python, Uses TF-IDF Vectorizer for cleaning the data and GridSearchCV on XGBoost.
    It has an accuracy of 75.05% .
             
    Dataset is of 724.5mb scrapped from Kaggle.
             
    Upload a resume or try the sample to see it in action.
    """)

@st.cache_data
def load_model():
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    mdl = joblib.load("models/xgb_best.pkl")
    lbl = joblib.load("models/label_encoder.pkl")
    return vec, mdl, lbl

def get_sample_resumes():
    sample_folder = "samples"
    sample_files = [
        f for f in os.listdir(sample_folder)
        if f.endswith(".pdf")
    ]
    sample_paths = [os.path.join(sample_folder, f) for f in sample_files]
    return sample_files, sample_paths

uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"])
if uploaded is not None:
    resume_bytes = uploaded.read()  
    text = extract_text(resume_bytes)
    st.text_area("Extracted Text:", text, height=300)

sample_files, sample_paths = get_sample_resumes()
selected_sample = None
if sample_files:
    selected_sample = st.selectbox("Or select a sample resume to try:", sample_files)
    sample_clicked = st.button("Try Selected Sample Resume")
else:
    sample_clicked = False

resume_bytes = None
if uploaded:
    resume_bytes = uploaded
elif sample_clicked and selected_sample:
    idx = sample_files.index(selected_sample)
    sample_path = sample_paths[idx]
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            resume_bytes = f.read()
        st.info(f"Loaded sample resume: {selected_sample}")
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