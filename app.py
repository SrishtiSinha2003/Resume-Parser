import streamlit as st
import PyPDF2
from docx import Document
from utils import calculate_hybrid_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ResumeLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #08111f;
    --panel: #0f1b2d;
    --panel-2: #132238;
    --border: #243650;
    --text: #f1f5f9;
    --muted: #91a3b8;
    --accent: #38bdf8;
    --accent-soft: rgba(56,189,248,.12);
    --success: #34d399;
    --success-soft: rgba(52,211,153,.10);
    --warning: #fbbf24;
    --warning-soft: rgba(251,191,36,.10);
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding: 1.5rem 2.2rem 3rem;
}

/* Remove unnecessary Streamlit chrome spacing */
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #091525;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Brand */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 3px;
}

.brand-mark {
    width: 38px;
    height: 38px;
    border: 1px solid #38bdf8;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    font-size: 22px;
    font-weight: 800;
    background: rgba(56,189,248,.08);
}

.brand-name {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1px;
}

.subtitle {
    color: var(--muted);
    font-size: 14px;
    margin: 0 0 25px 50px;
}

/* Section headings */
.section-title {
    font-size: 21px;
    font-weight: 700;
    color: var(--text);
    margin: 8px 0 15px;
}

.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 6px;
}

/* Input cards */
.input-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    min-height: 270px;
}

.input-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}

.input-help {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 16px;
}

/* Score cards */
.metric {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    min-height: 145px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.metric-value {
    font-size: 38px;
    line-height: 1.1;
    font-weight: 800;
    color: var(--text);
    margin: 9px 0 5px;
}

.metric-note {
    color: var(--muted);
    font-size: 12px;
}

/* Result panels */
.result-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
}

.summary-box {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
}

/* Skill list */
.skill-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px 12px;
    margin-top: 10px;
}

.skill-item {
    border: 1px solid #1e3a4d;
    background: var(--success-soft);
    color: #a7f3d0;
    border-radius: 9px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 600;
}

.missing-item {
    border: 1px solid #59431a;
    background: var(--warning-soft);
    color: #fde68a;
    border-radius: 9px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    border-radius: 9px !important;
    min-height: 45px;
    font-weight: 700 !important;
    border: 1px solid #2d9bd0 !important;
    transition: .2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--accent) !important;
}

/* Upload */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #94a3b8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8;
}

/* Sidebar */
.sidebar-heading {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 18px;
}

.sidebar-label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin: 22px 0 8px;
}

.sidebar-feature {
    display: flex;
    justify-content: space-between;
    color: #cbd5e1;
    font-size: 13px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(36,54,80,.55);
}

.sidebar-feature span:last-child {
    color: #34d399;
    font-weight: 700;
}

.footer-note {
    color: #64748b;
    font-size: 12px;
    text-align: center;
    margin-top: 35px;
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

    if file_name.endswith(".docx"):
        document = Document(uploaded_file)
        return "\n".join(
            paragraph.text for paragraph in document.paragraphs
        )

    return ""

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="brand">
    <div class="brand-mark">R</div>
    <div class="brand-name">ResumeLens</div>
</div>
<div class="subtitle">
    AI-powered resume screening and job compatibility analysis
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-heading">ResumeLens</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Compare a resume with a target role using keyword, semantic, "
        "and technical-skill analysis."
    )

    st.markdown('<div class="sidebar-label">Analysis engine</div>',
                unsafe_allow_html=True)

    features = [
        ("Keyword similarity", "ON"),
        ("Semantic matching", "ON"),
        ("Technical skills", "ON"),
        ("Document analysis", "ON"),
    ]

    for label, status in features:
        st.markdown(
            f'<div class="sidebar-feature"><span>{label}</span>'
            f'<span>{status}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-label">Supported files</div>',
                unsafe_allow_html=True)

    st.caption("PDF and DOCX resumes")

    st.markdown(
        '<div class="footer-note">ResumeLens • NLP-powered screening</div>',
        unsafe_allow_html=True
    )

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="eyebrow">Start analysis</div>',
            unsafe_allow_html=True)
st.markdown('<div class="section-title">Provide your resume and target role</div>',
            unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="input-card">
        <div class="input-title">Resume</div>
        <div class="input-help">
            Upload the PDF or DOCX you want to evaluate.
        </div>
    """, unsafe_allow_html=True)

    resume_file = st.file_uploader(
        "Choose a resume",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX"
    )

    if resume_file:
        st.success(f"Ready: {resume_file.name}")
    else:
        st.caption("No resume selected yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="input-card">
        <div class="input-title">Job description</div>
        <div class="input-help">
            Paste the complete description, including required skills.
        </div>
    """, unsafe_allow_html=True)

    job_description = st.text_area(
        "Job description",
        height=180,
        label_visibility="collapsed",
        placeholder=(
            "Example:\n"
            "Java Backend Developer\n\n"
            "Required skills: Java, Spring Boot, SQL...\n"
            "Responsibilities: Build REST APIs..."
        )
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "Analyze compatibility",
    type="primary",
    use_container_width=True
)

# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if resume_file is None:
        st.error("Please upload a resume.")
        st.stop()

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    with st.spinner("Extracting resume text..."):
        resume_text = extract_text(resume_file)

    if not resume_text.strip():
        st.error("Could not extract text from the resume.")
        st.stop()

    with st.spinner("Running resume compatibility analysis..."):
        result = calculate_hybrid_score(
            resume_text,
            job_description
        )

    st.success("Analysis complete.")

    # =====================================================
    # RESULTS HEADER
    # =====================================================

    st.markdown("---")
    st.markdown('<div class="eyebrow">Analysis result</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Resume compatibility</div>',
        unsafe_allow_html=True
    )

    ats_score = result["ats_score"]
    skill_score = result["skill_score"]
    semantic_score = result["semantic_score"]
    tfidf_score = result["tfidf_score"]

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">ATS compatibility</div>
            <div class="metric-value">{ats_score}</div>
            <div class="metric-note">Overall score out of 100</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Skill coverage</div>
            <div class="metric-value">{skill_score}%</div>
            <div class="metric-note">Required technical skills matched</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Semantic match</div>
            <div class="metric-value">{semantic_score}%</div>
            <div class="metric-note">Meaning-based document similarity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.progress(
        max(0.0, min(float(ats_score) / 100, 1.0))
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    matched_skills = result["matched_skills"]
    missing_skills = result["missing_skills"]

    if missing_skills:
        summary = (
            f"The resume matches {len(matched_skills)} identified job skills "
            f"and is missing {len(missing_skills)}. Review the missing-skill "
            f"list below to identify areas that may need stronger evidence "
            f"in the resume."
        )
    else:
        summary = (
            f"The resume matches all {len(matched_skills)} identified "
            f"technical skills from the job description."
        )

    st.markdown(
        f'<div class="summary-box"><strong>Match summary:</strong> '
        f'{summary}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Skills",
        "Match details",
        "Resume"
    ])

    # =====================================================
    # OVERVIEW
    # =====================================================

    with tab1:
        st.markdown("### Score composition")

        score_data = {
            "TF-IDF similarity": tfidf_score,
            "Semantic similarity": semantic_score,
            "Skill coverage": skill_score
        }

        for name, score in score_data.items():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.caption(name)
                st.progress(
                    max(0.0, min(float(score) / 100, 1.0))
                )
            with col_b:
                st.markdown(f"**{score}%**")

        st.info(
            "These scores are outputs of the project's hybrid scoring "
            "method and are not a validated prediction of hiring decisions."
        )

    # =====================================================
    # SKILLS
    # =====================================================

    with tab2:
        skill_col1, skill_col2 = st.columns(2, gap="large")

        with skill_col1:
            st.markdown("### Matched skills")

            if matched_skills:
                html = '<div class="skill-grid">'
                for skill in matched_skills:
                    html += (
                        f'<div class="skill-item">✓ '
                        f'{skill.title()}</div>'
                    )
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No matching skills detected.")

        with skill_col2:
            st.markdown("### Skills to strengthen")

            if missing_skills:
                html = '<div class="skill-grid">'
                for skill in missing_skills:
                    html += (
                        f'<div class="missing-item">○ '
                        f'{skill.title()}</div>'
                    )
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.success("No missing technical skills detected.")

    # =====================================================
    # MATCH DETAILS
    # =====================================================

    with tab3:
        st.markdown("### Semantic skill matching")

        for item in result["match_details"]:
            with st.container(border=True):
                a, b, c = st.columns([2, 2, 1])

                with a:
                    st.caption("JOB SKILL")
                    st.write(item["job_skill"].title())

                with b:
                    st.caption("BEST RESUME MATCH")
                    st.write(item["resume_match"].title())

                with c:
                    st.caption("SIMILARITY")
                    st.write(item["similarity"])

    # =====================================================
    # RESUME
    # =====================================================

    with tab4:
        st.markdown("### Extracted resume text")

        st.text_area(
            "Resume content",
            resume_text,
            height=450,
            label_visibility="collapsed"
        )

        st.download_button(
            "Download extracted text",
            data=resume_text,
            file_name="extracted_resume.txt",
            mime="text/plain",
            use_container_width=True
        )

        with st.expander("View extracted technical phrases"):
            st.markdown("**Job description skills**")
            st.write(result["job_phrases"])

            st.markdown("**Resume skills**")
            st.write(result["resume_phrases"])

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    '<div class="footer-note">'
    'ResumeLens • Python • Streamlit • NLP • Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
