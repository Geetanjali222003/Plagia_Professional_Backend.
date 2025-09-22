import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
# from ddgs import DDGS
from duckduckgo_search import DDGS


app = FastAPI()

# Global variables (loaded at startup)
summarizer = None
embedder = None


@app.on_event("startup")
async def load_models():
    """Load heavy models once at startup"""
    global summarizer, embedder
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Models loaded and ready!")


def extract_text(file: UploadFile):
    """Extract text from PDF or DOCX"""
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text("text")
    elif file.filename.endswith(".docx"):
        from docx import Document
        doc = Document(file.file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
    return text


@app.post("/check_plagiarism/")
async def check_plagiarism(file: UploadFile = File(...)):
    try:
        # Step 1: Extract text
        text = extract_text(file)
        if not text.strip():
            return JSONResponse({"error": "File has no readable text"}, status_code=400)

        # Step 2: Summarize
        summary = summarizer(text[:2000], max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

        # Step 3: Search for plagiarism
        plagiarism_matches = []
        plagiarism_percent = 0
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        with DDGS() as ddgs:
            for chunk in chunks[:5]:  # limit to first 5 chunks for speed
                results = list(ddgs.text(chunk, max_results=3))
                for r in results:
                    snippet = r.get("body") or ""
                    sim = util.cos_sim(embedder.encode(chunk), embedder.encode(snippet)).item()
                    if sim > 0.7:
                        plagiarism_matches.append({
                            "source": r.get("href"),
                            "similarity": round(sim * 100, 2),
                            "snippet": snippet[:200]
                        })
                        plagiarism_percent += 10  # rough scoring

        return {
            "plagiarism_percent": min(plagiarism_percent, 100),
            "plagiarism_matches": plagiarism_matches,
            "summary": summary
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
