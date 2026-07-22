from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import re


app = FastAPI(
    title="SafeAnswer Grounded QA API",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

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
    "be", "as"
}


def tokenize(text: str):
    """
    Convert text into normalized tokens.
    """
    return set(
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS
    )


def sentence_split(text: str):
    """
    Split chunk text into sentences.
    """
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )
        if sentence.strip()
    ]


def score_sentence(question: str, sentence: str):
    """
    Calculate a conservative lexical grounding score.
    """

    question_tokens = tokenize(question)
    sentence_tokens = tokenize(sentence)

    if not question_tokens or not sentence_tokens:
        return 0.0

    overlap = question_tokens.intersection(sentence_tokens)

    coverage = len(overlap) / len(question_tokens)

    # Exact phrase match gets additional weight.
    question_clean = question.lower().strip(" ?.!")

    phrase_bonus = 0.0

    if (
        question_clean
        and question_clean in sentence.lower()
    ):
        phrase_bonus = 0.35

    return min(
        1.0,
        coverage + phrase_bonus
    )


# ---------------------------------------------------------
# Grounded QA Logic
# ---------------------------------------------------------

def answer_question(
    question: str,
    chunks: List[Chunk]
):

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question or not question.strip():

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Validate chunks
    # -----------------------------------------------------

    valid_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.chunk_id
            and chunk.chunk_id.strip()
            and chunk.text
            and chunk.text.strip()
        )
    ]


    if not valid_chunks:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Score every sentence in every chunk
    # -----------------------------------------------------

    candidates = []


    for chunk in valid_chunks:

        for sentence in sentence_split(chunk.text):

            score = score_sentence(
                question,
                sentence
            )

            if score > 0:

                candidates.append({
                    "score": score,
                    "sentence": sentence,
                    "chunk_id": chunk.chunk_id
                })


    # -----------------------------------------------------
    # No grounded evidence
    # -----------------------------------------------------

    if not candidates:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Sort by score
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["chunk_id"]
        )
    )


    best = candidates[0]


    # -----------------------------------------------------
    # Conservative answerability threshold
    # -----------------------------------------------------

    if best["score"] < 0.45:

        confidence = min(
            0.3,
            best["score"] * 0.5
        )

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=round(
                confidence,
                3
            ),
            answerable=False
        )


    # -----------------------------------------------------
    # Calculate confidence
    # -----------------------------------------------------

    confidence = min(
        0.98,
        0.55 + (
            0.43 * best["score"]
        )
    )


    # -----------------------------------------------------
    # Verify citation ID is real
    # -----------------------------------------------------

    valid_chunk_ids = {
        chunk.chunk_id
        for chunk in valid_chunks
    }


    citation = best["chunk_id"]


    if citation not in valid_chunk_ids:

        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )


    # -----------------------------------------------------
    # Return grounded answer
    # -----------------------------------------------------

    return QAResponse(
        answer=best["sentence"],
        citations=[citation],
        confidence=round(
            confidence,
            3
        ),
        answerable=True
    )


# ---------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------

@app.post(
    "/grounded-qa",
    response_model=QAResponse
)
def grounded_qa(request: QARequest):

    return answer_question(
        request.question,
        request.chunks
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "SafeAnswer Grounded QA API"
    }