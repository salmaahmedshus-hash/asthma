from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()


class Question(BaseModel):
    question: str


PDF_FILE = "asthma-diagnosis-monitoring-and-chronic-asthma-management-bts-nice-sign-pdf-66143958279109.pdf"


def load_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), PDF_FILE)

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


pdf_text = load_pdf()


chunks = []
chunk_size = 1500

for i in range(0, len(pdf_text), chunk_size):
    chunks.append(pdf_text[i:i + chunk_size])


vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(chunks)


@app.get("/")
def home():
    return {
        "message": "Asthma RAG API is working!"
    }


@app.post("/ask")
def ask_question(data: Question):

    question_vector = vectorizer.transform([data.question])

    similarities = cosine_similarity(
        question_vector,
        vectors
    )[0]

    best_indexes = similarities.argsort()[-3:][::-1]

    answers = []

    for index in best_indexes:
        if similarities[index] > 0:
            answers.append(chunks[index])

    if not answers:
        answer = "I could not find relevant information in the PDF."
    else:
        answer = "\n\n".join(answers)

    return {
        "question": data.question,
        "answer": answer
    }