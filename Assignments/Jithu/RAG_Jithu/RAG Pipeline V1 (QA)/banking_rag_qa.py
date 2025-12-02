from groq import Groq
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

model_emb = SentenceTransformer("all-MiniLM-L6-v2")

# ---- CONFIG ----
GROQ_API_KEY = ""


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

DOCUMENTS = {
    "KYC Policy": load_pdf("data/kyc_policy.pdf"),
    "AML Typologies": load_pdf("data/aml_typologies.pdf"),
    "Risk Rules": load_pdf("data/risk_rules.pdf"),
}

client = Groq(api_key=GROQ_API_KEY)

# ---- EMBEDDING FUNCTION ----
def get_embedding(text):
    return model_emb.encode(text).astype("float32")

# ---- BUILD VECTOR STORE ----
texts = list(DOCUMENTS.values())
keys = list(DOCUMENTS.keys())
print(DOCUMENTS)
embeddings = np.vstack([get_embedding(t) for t in texts])

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# ---- RETRIEVAL ----
def retrieve_relevant_text(question, k=1):
    q_emb = get_embedding(question).reshape(1, -1)
    _, idx = index.search(q_emb, k)
    return texts[idx[0][0]]


# ---- LLM RESPONSE ----
def ask(question):
    context = retrieve_relevant_text(question)

    prompt = f"""
You are a compliance assistant. ONLY use information from below context. 
If answer not found, say: "Not enough information."

CONTEXT:
{context}

QUESTION: {question}
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip()


# ---- DEMO ----
if __name__ == "__main__":
    print(ask("What triggers enhanced due diligence?"))
