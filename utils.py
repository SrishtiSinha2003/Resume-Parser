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
    "experience",
    "candidate",
    "candidates",
    "responsibility",
    "responsibilities",
    "requirement",
    "requirements",
    "role",
    "roles",
    "team",
    "teams",
    "company",
    "organization",
    "work",
    "working",
    "ability",
    "abilities",
    "knowledge",
    "environment",
    "project",
    "projects",
    "job",
    "position",
    "professional",
    "professionals",
    "skills",
    "skill",
    "developer",
    "developers",
    "engineer",
    "engineers",
    "employee",
    "employees",
    "candidate profile",
    "job description",
    # HR / job-posting noise
    "looking",
    "motivated",
    "join",
    "title",
    "description",
    "hybrid",
    "onsite",
    "remote",
    "location",
    "salary",
    "apply",
    "application",
    "degree",
    "bachelor",
    "master",
    "education",
    "field",
    "related field",
    "familiarity",
    "understanding",
    "strong",
    "good",
    "excellent",
    "plus",
    "bonus",
    "preferred",
    "required",
    "must",
    "clean",
    "maintainable",
    "reusable",
    "code",
    "design",
    "debugging",
    "optimize",
    "implement",
    "build",
    "develop",
    "write",
    "maintain",
}


# Named entity labels that are NOT skills
_NOISE_ENTITY_LABELS = {"GPE", "LOC", "PERSON", "ORG", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"}


def strip_noise_entities(text):
    """Remove location, person, org, and date entities from text before keyword extraction."""
    doc = nlp(text)
    tokens = []
    skip_until = -1
    for token in doc:
        if token.i < skip_until:
            continue
        ent = token.ent_type_
        if ent in _NOISE_ENTITY_LABELS:
            skip_until = token.i + 1
            continue
        tokens.append(token.text)
    return " ".join(tokens)


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

    if not phrase:
        return False

    words = phrase.split()

    # Ignore very long phrases
    if len(words) > 5:
        return False

    # Ignore single generic words
    if phrase in GENERIC_TERMS:
        return False

    # Ignore phrases containing only generic words
    meaningful_words = [
        word for word in words
        if word not in GENERIC_TERMS
    ]

    if not meaningful_words:
        return False

    # Ignore very short phrases
    if len(phrase) < 2:
        return False

    return True


# =========================================================
# 1. KEYBERT KEYWORD EXTRACTION
# =========================================================

def extract_keybert_phrases(text, top_n=20):
    """
    Extract important keywords/keyphrases using KeyBERT.
    """

    text = strip_noise_entities(text)

    keywords = kw_model.extract_keywords(
        text,
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
    """
    Extract noun phrases using spaCy.
    """

    doc = nlp(text)

    phrases = []

    for chunk in doc.noun_chunks:

        phrase = normalize_phrase(chunk.text)

        if is_valid_phrase(phrase):
            phrases.append(phrase)

    return phrases


# =========================================================
# COMBINE KEYBERT + SPACY
# =========================================================

def extract_candidate_phrases(text, top_n=20):
    """
    Combine KeyBERT and spaCy results.
    """

    keybert_phrases = extract_keybert_phrases(
        text,
        top_n=top_n
    )

    spacy_phrases = extract_spacy_phrases(text)

    # Combine both approaches
    combined = keybert_phrases + spacy_phrases

    # Remove duplicates
    unique_phrases = list(dict.fromkeys(combined))

    return unique_phrases


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