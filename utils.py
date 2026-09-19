import re
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# =========================================================
# LOAD MODELS
# =========================================================

nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# KeyBERT uses the same Sentence Transformer
kw_model = KeyBERT(model=embedding_model)


# =========================================================
# GENERIC WORDS / PHRASES TO IGNORE
# =========================================================

GENERIC_TERMS = {
    # People / HR
    "experience", "candidate", "candidates", "employee", "employees",
    "professional", "professionals", "team", "teams", "member",
    # Job posting boilerplate
    "responsibility", "responsibilities", "requirement", "requirements",
    "role", "roles", "position", "job", "title", "description",
    "candidate profile", "job description",
    # Company / org
    "company", "organization", "organization",
    # Work verbs / generic actions
    "work", "working", "looking", "motivated", "join", "apply",
    "implement", "build", "develop", "write", "maintain", "optimize",
    "design", "debug", "debugging", "test", "testing", "manage",
    "collaborate", "communicate", "ensure", "support", "provide",
    # Adjectives / qualifiers
    "ability", "abilities", "knowledge", "understanding", "familiarity",
    "strong", "good", "excellent", "clean", "maintainable", "reusable",
    "preferred", "required", "must", "plus", "bonus", "related",
    "scalable", "motivated", "relevant",
    # Location / logistics
    "hybrid", "onsite", "remote", "location", "salary",
    # Education
    "degree", "bachelor", "master", "education", "field", "related field",
    # Misc
    "environment", "project", "projects", "skill", "skills",
    "developer", "developers", "engineer", "engineers",
    "application", "applications", "code", "information", "technology",
    "information technology",
}

# Known tech skills that should always be kept even if POS looks odd
TECH_SKILL_ALLOWLIST = {
    # Programming Languages
    "java", "python", "c", "c++", "c#", "javascript",
    "typescript", "go", "golang", "rust", "kotlin", "scala",
    
    # Backend
    "spring", "spring boot", "spring security",
    "hibernate", "maven", "gradle",
    "node", "node.js", "node js",
    "express", "express.js", "express js",
    "django", "flask", "fastapi",
    "rest", "rest api", "restful", "restful api",
    "graphql", "grpc",
    "microservices", "microservices architecture",
    
    # Frontend
    "react", "react.js", "react js", "angular", "vue",
    "vue.js", "html", "css", "bootstrap", "tailwind",
    
    # Databases
    "sql", "mysql", "postgresql", "postgres",
    "mongodb", "mongo", "redis", "oracle",
    "elasticsearch",
    
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "azure", "gcp",
    "jenkins", "ci/cd", "ci", "cd",
    
    # Version Control
    "git", "github", "gitlab",
    
    # Security
    "jwt", "oauth", "oauth2",
    "authentication", "authorization",
    
    # Testing
    "unit testing", "unit tests", "junit",
    "pytest", "selenium",
    
    # CS
    "data structures",
    "algorithms",
    "data structures and algorithms",
    "oop",
    "solid",
    "object oriented programming",
    "operating systems",
    "computer networks",
    "dbms",
    "database management",
    
    # AI / ML
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "computer vision",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
    "transformers",
    "bert",
}


# Named entity labels that are NOT skills
_NOISE_ENTITY_LABELS = {"GPE", "LOC", "PERSON", "ORG", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"}

# POS tags allowed in a valid skill phrase
_ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "NUM"}

# POS tags that disqualify a phrase (verbs, pronouns, determiners, etc.)
_REJECT_POS = {"VERB", "PRON", "DET", "CONJ", "CCONJ", "SCONJ", "INTJ", "PUNCT", "SPACE"}


def strip_noise_entities(text):
    """Remove location, person, org, date, and numeric entities from text."""
    doc = nlp(text)
    remove_spans = [
        (ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.label_ in _NOISE_ENTITY_LABELS
    ]
    if not remove_spans:
        return text
    result, prev = [], 0
    for start, end in remove_spans:
        result.append(text[prev:start])
        prev = end
    result.append(text[prev:])
    return re.sub(r"\s+", " ", "".join(result)).strip()


def _pos_validate_phrase(phrase):
    """Return True if the phrase looks like a skill based on POS tags."""
    # Always allow known tech terms
    if phrase in TECH_SKILL_ALLOWLIST:
        return True
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ in _REJECT_POS:
            return False
    pos_tags = [t.pos_ for t in doc]
    # Must have at least one noun or proper noun
    return any(p in {"NOUN", "PROPN"} for p in pos_tags)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_phrase(text):
    text = text.lower()

    # Keep characters commonly used in technical skills
    # such as C++, C#, .NET, etc.
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# FILTER PHRASES
# =========================================================

def is_valid_phrase(phrase):
    phrase = normalize_phrase(phrase)

    if not phrase or len(phrase) < 2:
        return False

    words = phrase.split()

    # Ignore very long phrases
    if len(words) > 4:
        return False

    # Reject pronouns / articles anywhere in phrase
    _stop = {"we", "the", "our", "a", "an", "this", "that", "its", "their", "your", "i", "you"}
    if any(w in _stop for w in words):
        return False

    # Exact match against generic terms
    if phrase in GENERIC_TERMS:
        return False

    # All words are generic
    meaningful = [w for w in words if w not in GENERIC_TERMS]
    if not meaningful:
        return False

    # Check allowlist before expensive POS check
    if phrase in TECH_SKILL_ALLOWLIST:
        return True

    # POS-based validation
    return _pos_validate_phrase(phrase)


# =========================================================
# 1. KEYBERT KEYWORD EXTRACTION
# =========================================================

def extract_keybert_phrases(text, top_n=20):
    cleaned = strip_noise_entities(text)

    keywords = kw_model.extract_keywords(
        cleaned,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )

    phrases = []
    for phrase, score in keywords:
        phrase = normalize_phrase(phrase)
        if is_valid_phrase(phrase):
            phrases.append(phrase)
    return phrases


# =========================================================
# 2. SPACY NOUN PHRASE EXTRACTION
# =========================================================

def extract_spacy_phrases(text):
    doc = nlp(text)
    noise_token_indices = {
        token.i
        for ent in doc.ents
        if ent.label_ in _NOISE_ENTITY_LABELS
        for token in ent
    }
    phrases = []
    for chunk in doc.noun_chunks:
        if any(token.i in noise_token_indices for token in chunk):
            continue
        # Skip chunks whose root is a generic/HR word
        if chunk.root.lemma_.lower() in GENERIC_TERMS:
            continue
        phrase = normalize_phrase(chunk.text)
        if is_valid_phrase(phrase):
            phrases.append(phrase)
    return phrases


# =========================================================
# COMBINE KEYBERT + SPACY
# =========================================================

def extract_candidate_phrases(text, top_n=20):
    """
    Extract only recognized technical skills.

    KeyBERT and spaCy are NOT used to decide whether
    something is a skill. The technical skill allowlist
    defines what qualifies as a skill.
    """

    text = normalize_phrase(text)

    found_skills = []

    # Check longer phrases first
    sorted_skills = sorted(
        TECH_SKILL_ALLOWLIST,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        # Handle special characters safely
        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(skill)
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(pattern, text):

            # Convert to readable display format
            display_skill = skill

            if display_skill not in found_skills:
                found_skills.append(display_skill)

    return found_skills

# =========================================================
# 3. TF-IDF SIMILARITY
# =========================================================

def calculate_tfidf_similarity(resume_text, job_text):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        vectors = vectorizer.fit_transform([
            resume_text,
            job_text
        ])

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(score)

    except ValueError:
        return 0.0


# =========================================================
# 4. DOCUMENT-LEVEL SEMANTIC SIMILARITY
# =========================================================

def calculate_semantic_similarity(
    resume_text,
    job_text
):

    embeddings = embedding_model.encode(
        [
            resume_text,
            job_text
        ],
        normalize_embeddings=True
    )

    score = np.dot(
        embeddings[0],
        embeddings[1]
    )

    # Keep score in [0, 1]
    score = max(0.0, min(1.0, float(score)))

    return score


# =========================================================
# 5. SEMANTIC SKILL MATCHING
# =========================================================

def match_phrases_semantically(
    job_phrases,
    resume_phrases,
    threshold=0.65
):
    """
    Compare every job phrase with resume phrases
    using Sentence Transformer embeddings.
    """

    if not job_phrases:

        return {
            "matched_skills": [],
            "missing_skills": [],
            "match_details": []
        }

    if not resume_phrases:

        return {
            "matched_skills": [],
            "missing_skills": job_phrases,
            "match_details": []
        }

    # Create embeddings
    job_embeddings = embedding_model.encode(
        job_phrases,
        normalize_embeddings=True
    )

    resume_embeddings = embedding_model.encode(
        resume_phrases,
        normalize_embeddings=True
    )

    # Similarity matrix
    similarity_matrix = cosine_similarity(
        job_embeddings,
        resume_embeddings
    )

    matched_skills = []
    missing_skills = []
    match_details = []

    for i, job_phrase in enumerate(job_phrases):

        similarities = similarity_matrix[i]

        best_index = np.argmax(similarities)

        best_score = similarities[best_index]

        best_resume_phrase = resume_phrases[best_index]

        if best_score >= threshold:

            matched_skills.append(job_phrase)

            match_details.append({
                "job_skill": job_phrase,
                "resume_match": best_resume_phrase,
                "similarity": round(
                    float(best_score),
                    3
                )
            })

        else:

            missing_skills.append(job_phrase)

            match_details.append({
                "job_skill": job_phrase,
                "resume_match": best_resume_phrase,
                "similarity": round(
                    float(best_score),
                    3
                )
            })

    return {
        "matched_skills": sorted(
            set(matched_skills)
        ),

        "missing_skills": sorted(
            set(missing_skills)
        ),

        "match_details": match_details
    }


# =========================================================
# 6. SKILL MATCH SCORE
# =========================================================

def calculate_skill_score(
    matched_skills,
    job_skills
):

    if not job_skills:
        return 0.0

    return len(
        matched_skills
    ) / len(
        job_skills
    )


# =========================================================
# 7. COMPLETE ATS ANALYSIS
# =========================================================

def calculate_hybrid_score(
    resume_text,
    job_text
):

    resume_text = normalize_text(
        resume_text
    )

    job_text = normalize_text(
        job_text
    )

    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------

    tfidf_score = calculate_tfidf_similarity(
        resume_text,
        job_text
    )

    # -----------------------------------------
    # Document semantic similarity
    # -----------------------------------------

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_text
    )

    # -----------------------------------------
    # Extract phrases from job description
    # -----------------------------------------

    job_phrases = extract_candidate_phrases(
        job_text,
        top_n=20
    )

    # -----------------------------------------
    # Extract phrases from resume
    # -----------------------------------------

    resume_phrases = extract_candidate_phrases(
        resume_text,
        top_n=30
    )

    # -----------------------------------------
    # Semantic phrase matching
    # -----------------------------------------

    skill_matching = match_phrases_semantically(
        job_phrases,
        resume_phrases,
        threshold=0.65
    )

    matched_skills = skill_matching[
        "matched_skills"
    ]

    missing_skills = skill_matching[
        "missing_skills"
    ]

    # -----------------------------------------
    # Skill coverage
    # -----------------------------------------

    skill_score = calculate_skill_score(
        matched_skills,
        job_phrases
    )

    # -----------------------------------------
    # UNIFIED ATS SCORE
    # -----------------------------------------

    ats_score = (
        (tfidf_score * 0.40) +
        (semantic_score * 0.30) +
        (skill_score * 0.30)
    ) * 100

    return {

        "ats_score": round(
            ats_score,
            2
        ),

        "tfidf_score": round(
            tfidf_score * 100,
            2
        ),

        "semantic_score": round(
            semantic_score * 100,
            2
        ),

        "skill_score": round(
            skill_score * 100,
            2
        ),

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "job_phrases": job_phrases,

        "resume_phrases": resume_phrases,

        "match_details":
            skill_matching["match_details"]
    }