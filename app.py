
import streamlit as st
import PyPDF2
from docx import Document
from utils import calculate_hybrid_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI ATS Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1120, #111827);
    color: #f8fafc;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1500px;
}

/* Header */

.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Cards */

.card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}

/* Score */

.score-card {
    background: linear-gradient(135deg, #1e293b, #312e81);
    border: 1px solid #6366f1;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}

.score-number {
    font-size: 52px;
    font-weight: 800;
    color: #a5b4fc;
}

.score-label {
    font-size: 15px;
    color: #cbd5e1;
}

/* Skill badges */

.skill-badge {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #059669;
    padding: 8px 14px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
    font-weight: 600;
}

.missing-badge {
    display: inline-block;
    background: #451a03;
    color: #fbbf24;
    border: 1px solid #b45309;
    padding: 8px 14px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
    font-weight: 600;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 12px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* Text area */

textarea {
    border-radius: 12px !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif file_name.endswith(".docx"):

        document = Document(uploaded_file)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    return ""


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI ATS Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume using NLP, semantic similarity, '
    'and intelligent skill matching.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Analyzer Settings")

    st.info(
        "Upload your resume and paste a job description "
        "to calculate an AI-assisted compatibility score."
    )

    st.markdown("---")

    st.markdown("### 🧠 AI Techniques")

    st.checkbox("TF-IDF", value=True, disabled=True)
    st.checkbox("KeyBERT", value=True, disabled=True)
    st.checkbox("spaCy", value=True, disabled=True)
    st.checkbox("Sentence Transformers", value=True, disabled=True)

    st.markdown("---")

    st.caption("AI ATS Analyzer • Resume Intelligence")


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns([1, 1], gap="large")

with col1:

    st.markdown(
        '<div class="card-title">📤 Upload Your Resume</div>',
        unsafe_allow_html=True
    )

    resume_file = st.file_uploader(
        "Choose your resume",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX"
    )

    if resume_file:

        st.success("Resume uploaded successfully!")

        st.write(f"📄 **File:** {resume_file.name}")

        st.write(
            f"📦 **Size:** {resume_file.size / 1024:.1f} KB"
        )

    else:

        st.info("Upload a PDF or DOCX resume to continue.")


with col2:

    st.markdown(
        '<div class="card-title">💼 Job Description</div>',
        unsafe_allow_html=True
    )

    job_description = st.text_area(
        "Paste the complete job description",
        height=200,
        placeholder=(
            "Example: Software Developer at TCS...\n\n"
            "Required Skills:\n"
            "Python, SQL, Java, Data Structures..."
        )
    )

    st.caption(
        "💡 Tip: Include responsibilities and required skills "
        "for better matching."
    )


st.markdown("---")

# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
)

if analyze:

    if resume_file is None:

        st.error("Please upload a resume.")

        st.stop()

    if not job_description.strip():

        st.error("Please enter a job description.")

        st.stop()

    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    with st.spinner("📖 Extracting resume text..."):

        resume_text = extract_text(resume_file)

    if not resume_text.strip():

        st.error(
            "Could not extract text from the resume."
        )

        st.stop()

    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    with st.spinner(
        "🤖 Running AI-powered resume analysis..."
    ):

        result = calculate_hybrid_score(
            resume_text,
            job_description
        )

    st.success("✅ Resume analysis completed!")

    # =====================================================
    # SCORE SECTION
    # =====================================================

    st.markdown("## 🎯 Resume Compatibility Dashboard")

    ats_score = result["ats_score"]

    skill_score = result["skill_score"]

    semantic_score = result["semantic_score"]

    tfidf_score = result["tfidf_score"]

    # Score cards

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🎯 UNIFIED ATS SCORE</div>
                <div class="score-number">{ats_score}</div>
                <div class="score-label">out of 100</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🧠 SKILL COVERAGE</div>
                <div class="score-number">{skill_score}%</div>
                <div class="score-label">Matched job skills</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🔍 SEMANTIC SIMILARITY</div>
                <div class="score-number">{semantic_score}%</div>
                <div class="score-label">Meaning-based matching</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📈 Overall Score")

    st.progress(
        max(0.0, min(float(ats_score) / 100, 1.0))
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Score Breakdown",
        "🛠️ Skills Analysis",
        "🔍 Semantic Details",
        "📄 Resume Preview"
    ])

    # =====================================================
    # TAB 1: SCORE BREAKDOWN
    # =====================================================

    with tab1:

        st.markdown("### 📊 AI Scoring Components")

        score_data = {
            "TF-IDF": tfidf_score,
            "Semantic": semantic_score,
            "Skills": skill_score
        }

        for name, score in score_data.items():

            st.write(f"**{name}** — {score}%")

            st.progress(
                max(0.0, min(float(score) / 100, 1.0))
            )

        st.markdown("---")

        st.info(
            "Scores represent the outputs of your current "
            "hybrid scoring function. They are not a validated "
            "prediction of hiring decisions."
        )

    # =====================================================
    # TAB 2: SKILLS
    # =====================================================

    with tab2:

        st.markdown("### ✅ Matched Skills")

        matched_skills = result["matched_skills"]

        if matched_skills:

            html = ""

            for skill in matched_skills:

                html += (
                    f'<span class="skill-badge">'
                    f'✓ {skill.title()}'
                    f'</span>'
                )

            st.markdown(
                html,
                unsafe_allow_html=True
            )

        else:

            st.info("No matching skills detected.")

        st.markdown("---")

        st.markdown("### ⚠️ Missing Skills")

        missing_skills = result["missing_skills"]

        if missing_skills:

            html = ""

            for skill in missing_skills:

                html += (
                    f'<span class="missing-badge">'
                    f'! {skill.title()}'
                    f'</span>'
                )

            st.markdown(
                html,
                unsafe_allow_html=True
            )

        else:

            st.success("No missing skills detected.")

    # =====================================================
    # TAB 3: SEMANTIC DETAILS
    # =====================================================

    with tab3:

        st.markdown("### 🔍 Semantic Matching Details")

        for item in result["match_details"]:

            with st.container(border=True):

                st.markdown(
                    f"**Job Skill:** {item['job_skill']}"
                )

                st.markdown(
                    f"**Best Resume Match:** "
                    f"{item['resume_match']}"
                )

                st.markdown(
                    f"**Similarity:** {item['similarity']}"
                )

    # =====================================================
    # TAB 4: RESUME PREVIEW
    # =====================================================

    with tab4:

        st.markdown("### 📄 Extracted Resume Text")

        st.text_area(
            "Resume Content",
            resume_text,
            height=500
        )

        st.download_button(
            "⬇️ Download Extracted Text",
            data=resume_text,
            file_name="extracted_resume.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("🔎 View Extracted Phrases"):

            st.markdown("#### Job Description Phrases")

            st.write(result["job_phrases"])

            st.markdown("#### Resume Phrases")

            st.write(result["resume_phrases"])


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with Python • Streamlit • NLP • Machine Learning"
)