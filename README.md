# 📄 AI Resume Classifier 

A **Streamlit-based web application** that classifies resumes into pre-defined job categories using Machine Learning.  
It allows users to upload their resumes (in PDF format), extracts the text, and predicts the most suitable job role.

---

## 🚀 Features
- **Upload Resume** — Supports .pdf formats.
- **Text Extraction** — Reads resume content automatically.
- **ML Classification** — Predicts job role/category.
- **Instant Results** — Displays predicted category in real time.
- **Deployed Online** — Access it directly via Streamlit Cloud.

---

## 🌐 Live Demo
[**Click here to try the app**](https://resume-classifier-tushar04-master.streamlit.app)  

---

## 🛠️ Tech Stack
- **Frontend & Hosting:** [Streamlit](https://streamlit.io/)
- **Backend/Processing:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy Matplotlib
- **Text Extraction:** PyPDF2 / python-docx
- **Model:** GridSearchCV on XGBOOST

---

## 📥 Installation & Local Setup

### Clone the repository

git clone https://github.com/Tushar04-Master/resume-classifier.git
cd resume-classifier

### Create a virtual environment ( Optional but recommended )
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### Install dependencies
pip install -r requirements.txt

### Run the Streamlit app
streamlit run app.py

## Project Structure
resume-classifer
    ->
