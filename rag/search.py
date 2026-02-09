import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voice.tts import speak
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

reader = PdfReader("data/manual.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(chunks, embeddings)

print("Manual ready. Ask questions!\n")
def keyword_search(chunks, query, k=3):
    results = []
    q_words = query.lower().split()

    for chunk in chunks:
        score = sum(word in chunk.lower() for word in q_words)
        if score > 0:
            results.append((score, chunk))

    results.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in results[:k]]

while True:
    query = input("Ask: ")
    speak("Let me check that for you.")

    partial_query = " ".join(query.split()[:3])

    print("\n[Prefetching using partial query...]\n")

    prefetch_docs = vectorstore.similarity_search(partial_query, k=3)
    vector_docs = vectorstore.similarity_search(query, k=3)

    keyword_docs = keyword_search(chunks, query, k=3)

    combined_docs = (
    prefetch_docs
    + vector_docs
    + [type("Doc", (), {"page_content": d}) for d in keyword_docs]
)

    context = "\n".join(
    list(dict.fromkeys([d.page_content for d in combined_docs]))
)   
    

    prompt = f"""
You are a voice technical support agent.

Answer using spoken English:
- Use short sentences.
- Use simple words.
- Give clear step-by-step actions.
- Avoid long explanations.
- Do not ask many questions.
- Give actions first, then optional checks.
- Max 12 words per sentence.

Context:
{context}

User Question:
{query}

Spoken Answer:
"""

    
    


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response.choices[0].message.content or ""
    print("\nAI Answer:\n", answer, "\n")
    speak(answer)
