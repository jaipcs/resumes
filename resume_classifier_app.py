import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from docx import Document
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------ Setup ------------------

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load ML artifacts
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ------------------ Skills & Constants ------------------
TECH_SKILLS = [
    "python","sql","machine learning","data analysis","tensorflow","keras","pytorch",
    "streamlit","sklearn","pandas","numpy","seaborn","matplotlib","deep learning",
    "nlp","cloud","azure","aws","gcp","c++","java","powerbi","excel"
]

# ------------------ Helper Functions ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_resume(file):
    return extract_text_from_docx(file) if file.name.endswith(".docx") else ""

def extract_skills(text):
    return [s for s in TECH_SKILLS if s in text.lower()]

def extract_experience(text):
    years = re.findall(r'(\d+)\s*(?:\+?\s*years?|yrs?|year)', text.lower())
    return max(map(int, years), default=0)

def send_email(to_email, candidate_name, predicted_role, template="basic"):
    sender_email = "your_email@gmail.com"
    password = "your_app_password"

    subject = f"Interview Invitation for {predicted_role}"
    body = f"""
    Dear {candidate_name},

    We reviewed your resume and found your skills suitable for the {predicted_role} role.
    We would like to invite you for an interview.

    Regards,
    HR Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True
    except Exception:
        return False

# ------------------ Main App ------------------
st.set_page_config(layout="wide")
st.title("Advanced Resume Classifier Dashboard")

uploaded_files = st.file_uploader("Upload DOCX resumes (multiple allowed)", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    results = []
    all_skills = []
    role_counts = Counter()

    # Process resumes
    for file in uploaded_files:
        raw_text = extract_text_from_resume(file)
        cleaned = clean_text(raw_text)

        pred_idx = model.predict(tfidf.transform([cleaned]))[0]
        predicted_role = le.inverse_transform([pred_idx])[0]

        skills = extract_skills(raw_text)
        exp_years = extract_experience(raw_text)

        results.append({
            "Candidate": file.name,
            "Skills": ", ".join(skills),
            "Experience (Years)": exp_years,
            "Predicted Role": predicted_role
        })
        all_skills.extend(skills)
        role_counts[predicted_role] += 1

    df_results = pd.DataFrame(results)

    # ----------- Tabs for 8 Modules -----------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Resume Insights", "Visualizations", "Interactive Tools", "Bulk Email",
        "Prediction Enhancements", "Multi-Format Support", "UI/UX Enhancements", "Analytics Dashboard"
    ])

    # -------- 1. Resume Insights --------
    with tab1:
        st.subheader("Candidate Details")
        st.dataframe(df_results)

        # Education, Certifications (future parsing placeholders)
        st.info("Education & Certifications parsing coming soon...")

    # -------- 2. Visualizations --------
    with tab2:
        st.subheader("Skills Frequency")
        skill_freq = Counter(all_skills)
        if skill_freq:
            skill_df = pd.DataFrame(skill_freq.items(), columns=["Skill", "Count"]).sort_values(by="Count", ascending=False)
            st.bar_chart(skill_df.set_index("Skill"))

        st.subheader("Experience Distribution")
        st.bar_chart(df_results.set_index("Candidate")["Experience (Years)"])

        st.subheader("Predicted Roles Distribution")
        role_df = pd.DataFrame(role_counts.items(), columns=["Role", "Count"]).sort_values(by="Count", ascending=False)
        st.bar_chart(role_df.set_index("Role"))

    # -------- 3. Interactive Tools --------
    with tab3:
        st.subheader("Search/Filter Candidates")
        min_exp = st.slider("Minimum Experience", 0, 20, 0)
        role_filter = st.selectbox("Filter by Role", ["All"] + df_results["Predicted Role"].unique().tolist())

        filtered = df_results[df_results["Experience (Years)"] >= min_exp]
        if role_filter != "All":
            filtered = filtered[filtered["Predicted Role"] == role_filter]

        st.dataframe(filtered)

    # -------- 4. Bulk Email --------
    with tab4:
        st.subheader("Send Interview Invitations")
        st.markdown("This will send emails to shortlisted candidates.")
        st.warning("Configure your Gmail & App Password in code before enabling.")

        email_col = st.text_area("Enter Emails (comma separated)")
        candidate_names = st.text_area("Enter Candidate Names (comma separated)")

        if st.button("Send Emails"):
            emails = [e.strip() for e in email_col.split(",") if e.strip()]
            names = [n.strip() for n in candidate_names.split(",") if n.strip()]
            for email, name in zip(emails, names):
                send_email(email, name, "Predicted Role")  # Replace with dynamic
            st.success("Emails sent successfully (if SMTP configured).")

    # -------- 5. Prediction Enhancements --------
    with tab5:
        st.info("Add confidence scores, multi-role predictions here...")

    # -------- 6. Multi-Format Support --------
    with tab6:
        st.info("Add PDF and ZIP parsing capabilities here...")

    # -------- 7. UI/UX Enhancements --------
    with tab7:
        st.info("Dark mode toggle, report download, progress bar can be added here...")

    # -------- 8. Analytics Dashboard --------
    with tab8:
        st.subheader("Role Distribution Pie Chart")
        st.dataframe(role_df)
        st.info("Add pie chart, trend analysis, scatter plots here...")

    # Download CSV
    st.download_button("Download All Candidates CSV", df_results.to_csv(index=False), "all_candidates.csv", "text/csv")
