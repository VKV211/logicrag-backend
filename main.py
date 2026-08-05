import logging
import os
import shutil

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_pipeline import process_and_index_pdf, retrieve_context, index_exists, UPLOAD_DIR
from classifier import classify_question
from prompts import build_final_prompt
from memory import add_turn, get_history_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logicrag")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file (or your Render environment variables).")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="LogicRAG Backend")

# allow the Streamlit frontend (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Global safety net ----------------
# Catches ANY unhandled exception anywhere in the app and returns a clean,
# readable JSON error instead of a raw 500 with no message. The full traceback
# still gets logged on the server for debugging.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Something went wrong on the server while processing your request. "
                "This has been logged. Please try again, and if it keeps happening, "
                "try a different file or question."
            )
        },
    )


class AskRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported. Please upload a .pdf file.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Could not save the uploaded file: {e}")

    try:
        num_chunks = process_and_index_pdf(save_path, file.filename)
    except ValueError as e:
        # Expected, user-facing issue (e.g. empty/scanned PDF)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error while indexing PDF")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to process '{file.filename}'. This can happen with corrupted, "
                f"password-protected, or unusually formatted PDFs. Details: {e}"
            ),
        )

    return {
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "message": "PDF processed and added to the knowledge base.",
    }


@app.post("/ask")
async def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a PDF first before asking questions.",
        )

    # Step 1: classify reasoning type
    try:
        reasoning_type = classify_question(question)
    except Exception as e:
        logger.exception("Classification step failed")
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the AI model to classify your question. Please try again shortly. ({e})",
        )

    # Step 2: retrieve relevant chunks
    try:
        context, sources = retrieve_context(question)
    except Exception as e:
        logger.exception("Retrieval step failed")
        raise HTTPException(status_code=500, detail=f"Failed to search the document index. ({e})")

    # Step 3: build final prompt with history
    history = get_history_text()
    final_prompt = build_final_prompt(reasoning_type, context, history, question)

    # Step 4: call Gemini for the final answer
    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(final_prompt)
        answer = response.text.strip()
    except Exception as e:
        logger.exception("Gemini generation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not get a response from the AI model right now. "
                f"This is usually temporary — please try again in a moment. ({e})"
            ),
        )

    # Step 5: update memory
    add_turn(question, answer)

    return {
        "answer": answer,
        "reasoning_category": reasoning_type,
        "sources": sources,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}