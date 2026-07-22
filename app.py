from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import re


app = FastAPI(
    title="SafeAnswer Grounded QA API",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATA MODELS
# =========================================================

class Chunk(BaseModel):
    chunk_id: str
    text: str


class QARequest(BaseModel):
    question: str
    chunks: List[Chunk] = Field(default_factory=list)


class QAResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool


# =========================================================
# TEXT PROCESSING
# =========================================================

STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "what", "when", "where", "who",
    "why", "how", "which",
    "of", "in", "on", "for",
    "to", "and", "or",
    "with", "from", "by",
    "does", "do", "did",
    "it", "this", "that",
    "be", "as",
    "tell", "me"
}


def tokenize(text: str):

    return set(
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower()
        )
        if token not in STOPWORDS
    )


def sentence_split(text: str):

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )
        if sentence.strip()
    ]


# =========================================================
# QUESTION TYPE DETECTION
# =========================================================

def detect_question_type(question: str):

    q = question.lower().strip()

    if q.startswith("who"):
        return "who"

    if q.startswith("when") or "what year" in q:
        return "when"

    if q.startswith("where"):
        return "where"

    if q.startswith("why"):
        return "why"

    if q.startswith("how many"):
        return "number"

    if q.startswith("how much"):
        return "number"

    if q.startswith("how"):
        return "how"

    if q.startswith("what"):
        return "what"

    return "general"


# =========================================================
# SENTENCE RELEVANCE
# =========================================================

def sentence_score(
    question: str,
    sentence: str
):

    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)

    if not q_tokens or not s_tokens:
        return 0.0

    overlap = q_tokens.intersection(
        s_tokens
    )

    coverage = (
        len(overlap)
        / len(q_tokens)
    )

    # Entity / number overlap
    q_numbers = set(
        re.findall(
            r"\b\d{4}\b",
            question
        )
    )

    s_numbers = set(
        re.findall(
            r"\b\d{4}\b",
            sentence
        )
    )

    number_bonus = 0.0

    if q_numbers.intersection(
        s_numbers
    ):
        number_bonus = 0.2

    # Exact phrase bonus
    phrase_bonus = 0.0

    q_clean = question.lower().strip(
        " ?!."
    )

    if (
        len(q_clean) > 5
        and q_clean in sentence.lower()
    ):
        phrase_bonus = 0.3

    score = (
        coverage
        + number_bonus
        + phrase_bonus
    )

    return min(
        1.0,
        score
    )


# =========================================================
# FIND SUPPORTING EVIDENCE
# =========================================================

def find_evidence(
    question: str,
    chunks: List[Chunk]
):

    evidence = []

    for chunk in chunks:

        sentences = sentence_split(
            chunk.text
        )

        for sentence in sentences:

            score = sentence_score(
                question,
                sentence
            )

            if score >= 0.25:

                evidence.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "sentence": sentence,
                        "score": score
                    }
                )

    evidence.sort(
        key=lambda x: (
            -x["score"],
            x["chunk_id"]
        )
    )

    return evidence


# =========================================================
# ANSWER CONSTRUCTION
# =========================================================

def build_answer(
    question: str,
    evidence
):

    if not evidence:
        return None, [], 0.0


    # -----------------------------------------------------
    # Select strong evidence
    # -----------------------------------------------------

    strong = [
        item
        for item in evidence
        if item["score"] >= 0.35
    ]


    if not strong:
        return None, [], 0.0


    # Maximum evidence sentences to use
    selected = strong[:5]


    # -----------------------------------------------------
    # Remove duplicate sentences
    # -----------------------------------------------------

    seen = set()
    unique = []

    for item in selected:

        key = item["sentence"].lower()

        if key not in seen:

            seen.add(key)

            unique.append(item)


    selected = unique


    if not selected:
        return None, [], 0.0


    # -----------------------------------------------------
    # Combine evidence
    # -----------------------------------------------------

    answer_parts = [
        item["sentence"]
        for item in selected
    ]


    answer = " ".join(
        answer_parts
    )


    # -----------------------------------------------------
    # Collect ALL supporting citations
    # -----------------------------------------------------

    citations = []

    for item in selected:

        chunk_id = item["chunk_id"]

        if chunk_id not in citations:

            citations.append(
                chunk_id
            )


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    scores = [
        item["score"]
        for item in selected
    ]

    average_score = (
        sum(scores)
        / len(scores)
    )


    # Strongest evidence
    max_score = max(scores)


    confidence = (
        0.5 * max_score
        + 0.5 * average_score
    )


    confidence = min(
        0.98,
        max(
            0.0,
            confidence
        )
    )


    return (
        answer,
        citations,
        confidence
    )


# =========================================================
# MAIN QA FUNCTION
# =========================================================

def answer_question(
    question: str,
    chunks: List[Chunk]
):

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if (
        not question
        or not question.strip()
    ):

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Validate chunks
    # -----------------------------------------------------

    valid_chunks = []

    seen_ids = set()

    for chunk in chunks:

        if (
            not chunk.chunk_id
            or not chunk.chunk_id.strip()
        ):
            continue

        if (
            not chunk.text
            or not chunk.text.strip()
        ):
            continue

        # Prevent duplicate IDs
        if chunk.chunk_id in seen_ids:
            continue

        seen_ids.add(
            chunk.chunk_id
        )

        valid_chunks.append(
            chunk
        )


    if not valid_chunks:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Find evidence
    # -----------------------------------------------------

    evidence = find_evidence(
        question,
        valid_chunks
    )


    # -----------------------------------------------------
    # Build grounded answer
    # -----------------------------------------------------

    (
        answer,
        citations,
        confidence
    ) = build_answer(
        question,
        evidence
    )


    # -----------------------------------------------------
    # No sufficient evidence
    # -----------------------------------------------------

    if (
        not answer
        or not citations
    ):

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Validate every citation
    # -----------------------------------------------------

    valid_ids = {
        chunk.chunk_id
        for chunk in valid_chunks
    }


    citations = [
        citation
        for citation in citations
        if citation in valid_ids
    ]


    if not citations:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Enforce answerability threshold
    # -----------------------------------------------------

    if confidence < 0.45:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=min(
                0.3,
                round(
                    confidence,
                    3
                )
            ),
            answerable=False
        )


    # -----------------------------------------------------
    # Return grounded response
    # -----------------------------------------------------

    return QAResponse(
        answer=answer,
        citations=citations,
        confidence=round(
            confidence,
            3
        ),
        answerable=True
    )


# =========================================================
# API ENDPOINT
# =========================================================

@app.post(
    "/grounded-qa",
    response_model=QAResponse
)
def grounded_qa(
    request: QARequest
):

    return answer_question(
        request.question,
        request.chunks
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "SafeAnswer Grounded QA API",
        "version": "2.0.0"
    }