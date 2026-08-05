"""
Classifies a user question into one reasoning category using Gemini.
"""

import google.generativeai as genai

VALID_CATEGORIES = ["Mathematical", "Legal", "Medical", "Logical", "Comparative", "General"]

CLASSIFY_PROMPT = """Classify the following question into EXACTLY ONE of these categories:
Mathematical, Legal, Medical, Logical, Comparative, General

Use "General" for simple factual lookups, definitions, titles, names, or anything that
doesn't specifically require mathematical, legal, medical, comparative, or causal reasoning.

Respond with ONLY the category word, nothing else.

Question: {question}
Category:"""


def classify_question(question: str, model_name: str = "gemini-3.6-flash") -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(CLASSIFY_PROMPT.format(question=question))
    category = response.text.strip()

    # Guard against Gemini returning extra text/formatting
    for valid in VALID_CATEGORIES:
        if valid.lower() in category.lower():
            return valid

    return "General"  # safe fallback