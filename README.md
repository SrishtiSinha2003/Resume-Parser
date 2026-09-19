# 🔎 ResumeLens — AI Resume Screening & Job Matching

> **Turn a resume and a job description into an actionable compatibility report.**

ResumeLens is an NLP-powered resume screening and job-matching web application built with **Python and Streamlit**. It compares a candidate's resume against a target job description and presents an ATS-style compatibility score, technical skill coverage, semantic similarity, matched skills, missing skills, and detailed matching information.

<p align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://resume-parser-jhpugub4fg2n4pthpcwkbj.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SrishtiSinha2003/Resume-Parser)

</p>

---

## ✨ What does ResumeLens do?

ResumeLens answers a simple question:

> **"How closely does this resume match this job?"**

Upload a resume in **PDF/DOCX** format, paste a job description, and the application analyzes both documents using multiple NLP signals.

### 📊 The dashboard provides

- 🎯 **Unified ATS Score** — overall resume-job compatibility
- 🧠 **Skill Coverage** — percentage of identified job skills found in the resume
- 🔍 **Semantic Similarity** — meaning-based similarity between resume and job description
- ✅ **Matched Skills** — technical skills detected in both sides
- ⚠️ **Missing Skills** — technical skills present in the job description but absent from the resume
- 📈 **Score Breakdown** — individual components contributing to the analysis
- 🔬 **Semantic Match Details** — closest resume skill for each job skill
- 📄 **Resume Preview** — extracted resume text for verification

---

## 🧠 How the analysis works

ResumeLens uses a **hybrid NLP approach** rather than relying on a single similarity score.

```text
                  ┌─────────────────┐
                  │     RESUME      │
                  └────────┬────────┘
                           │
                    Text Extraction
                           │
                           ▼
                  ┌─────────────────┐
                  │  Preprocessing  │
                  └────────┬────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  NLP Feature Analysis  │
              └───────────┬────────────┘
                          / \
                         /   \
                        ▼     ▼
                   TF-IDF   Semantic
                           Embeddings
                        \     /
                         \   /
                          ▼ ▼
                   Skill Matching
                          │
                          ▼
                ┌───────────────────┐
                │ Compatibility     │
                │ Analysis          │
                └─────────┬─────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          ATS Score   Matched Skills  Missing Skills
```

### Core techniques

**TF-IDF + Cosine Similarity**  
Measures lexical similarity between the resume and job description.

**Sentence Transformers**  
Creates semantic embeddings so the system can compare text based on meaning rather than only exact word overlap.

**Technical Skill Extraction**  
Identifies relevant technical skills before performing skill-level matching.

**Semantic Skill Matching**  
Compares job skills against resume skills using embedding similarity to identify the closest matches.

---

## 🖥️ Interface

The application provides a focused ATS-style workflow:

### 1. Upload resume

Supported formats:

- PDF
- DOCX

### 2. Add job description

Paste the complete job description, including responsibilities and required/preferred skills.

### 3. Analyze

ResumeLens processes the documents and generates the compatibility report.

### 4. Explore the results

The dashboard separates the analysis into:

| Section | What it shows |
|---|---|
| **Overview** | Overall score and score components |
| **Skills** | Matched and missing technical skills |
| **Match Details** | Semantic job-skill → resume-skill matches |
| **Resume** | Extracted resume text and technical phrases |

---

## 🛠️ Tech Stack

### Application

- **Python**
- **Streamlit**

### NLP / Machine Learning

- **scikit-learn**
- **TF-IDF**
- **Cosine Similarity**
- **KeyBERT**
- **spaCy**
- **Sentence Transformers**
- **PyTorch / TorchVision**
- **Transformers**

### Document Processing

- **PyPDF2**
- **python-docx**

### Deployment

- **Streamlit Community Cloud**

---

## 📁 Project Structure

```text
Resume-Parser/
│
├── app.py                 # Streamlit application & UI
├── utils.py               # NLP processing and scoring logic
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignored files
└── README.md              # Project documentation
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SrishtiSinha2003/Resume-Parser.git
cd Resume-Parser
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Live Demo

### 🚀 Try ResumeLens

**Live application:**  
https://resume-parser-jhpugub4fg2n4pthpcwkbj.streamlit.app/

### 💻 Source Code

**GitHub repository:**  
https://github.com/SrishtiSinha2003/Resume-Parser

---

## 🧪 Example Workflow

A typical analysis looks like:

```text
Resume
  +
Job Description
  ↓
ResumeLens
  ↓
┌──────────────────────────────┐
│ ATS Compatibility     60.52  │
│ Skill Coverage         81.82%│
│ Semantic Similarity    74.66%│
└──────────────────────────────┘

Matched:
✓ Java
✓ Python
✓ Spring Boot
✓ MySQL
✓ Git
✓ REST API
✓ React

Missing:
! JWT
! Microservices
! Unit Testing
```

> Scores are generated by the project's NLP pipeline and should be treated as an analytical aid rather than a prediction of hiring decisions.

---

## 🎯 Use Cases

ResumeLens can be used for:

- 👩‍💻 **Candidates** — identify gaps between a resume and a target role
- 🧑‍💼 **Recruitment prototypes** — demonstrate automated resume screening
- 🎓 **Academic projects** — showcase NLP and machine-learning concepts
- 📚 **Learning** — understand practical applications of text similarity and semantic matching
- 🛠️ **Portfolio projects** — demonstrate an end-to-end NLP web application

---

## 🔮 Future Improvements

Possible extensions include:

- 📑 Section-wise resume scoring
- 🏆 Multi-resume ranking against one job description
- 📊 More detailed experience and education analysis
- 📝 Automated resume improvement suggestions
- 📥 Downloadable analysis reports
- 🌍 Multilingual resume/JD analysis
- ☁️ Persistent storage for analysis history
- 🔐 User authentication and private dashboards

---

## ⚠️ Disclaimer

ResumeLens is an educational and analytical tool. Its compatibility score is generated from NLP-based text and skill analysis and **does not represent an actual ATS score or guarantee interview selection**.

---

## 👩‍💻 Author

**Srishti Sinha**

Computer Science & Engineering

---

<p align="center">

### ⭐ If you find the project interesting, consider starring the repository!

**Built with Python • Streamlit • NLP • Machine Learning**

</p>
