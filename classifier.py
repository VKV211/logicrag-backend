"""
Classifies a user question into one reasoning category using Gemini.
"""
import os
from google import genai

VALID_CATEGORIES = ["Mathematical", "Legal", "Medical", "Logical", "Comparative", "General"]

CLASSIFY_PROMPT = """Classify the following question into EXACTLY ONE of these categories:
Mathematical, Legal, Medical, Logical, Comparative, General

Use "General" for simple factual lookups, definitions, titles, names, or anything that
doesn't specifically require mathematical, legal, medical, comparative, or causal reasoning.

Respond with ONLY the category word, nothing else.

Question: {question}
Category:"""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def classify_question(question: str, model_name: str = "gemini-2.5-flash") -> str:
    response = get_client().models.generate_content(
        model=model_name,
        contents=CLASSIFY_PROMPT.format(question=question),
    )
    category = response.text.strip()

    # Guard against Gemini returning extra text/formatting
    for valid in VALID_CATEGORIES:
        if valid.lower() in category.lower():
            return valid
    return "General"  # safe fallback
