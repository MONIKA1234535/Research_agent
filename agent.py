from pathlib import Path
import os

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_chunk_sources(folder_path):
    """Read .txt files from a folder and split each into paragraph-level chunks."""
    chunks = []
    folder = Path(folder_path)

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append({"text": paragraph, "source": file_path.name})

    return chunks


def retrieve_top_chunks(question, chunks, top_k=3):
    """Return the top_k chunks most similar to the question by TF-IDF cosine similarity."""
    texts = [chunk["text"] for chunk in chunks]
    tfidf_matrix = TfidfVectorizer().fit_transform([question] + texts)

    question_vector = tfidf_matrix[0:1]
    chunk_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(question_vector, chunk_vectors).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        result = chunks[idx].copy()
        result["score"] = float(scores[idx])
        results.append(result)

    return results


def build_prompt(question, top_chunks):
    """Format retrieved chunks and the question into a prompt for the LLM."""
    source_blocks = []
    for i, chunk in enumerate(top_chunks, start=1):
        source_blocks.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )

    sources_section = "\n\n".join(source_blocks)

    return (
        f"{sources_section}\n\n"
        f"Question: {question}\n\n"
        "You are a research assistant. Answer the question using ONLY the numbered"
        "sources below. For every claim, cite the source number in brackets, e.g. [Source 2]."
        'If the sources do not contain enough information to answer, say exactly: '
        ' "The provided sources do not contain enough information to answer this question." '
        "Do not use outside knowledge."
    )


def ask_llm(prompt):
    """Send a prompt to Groq and return the model's response text."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in .env in the project folder."
        )
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    chunks = load_and_chunk_sources("sources")

    while True:
        question = input("\nEnter a question (or 'quit' to exit): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        top_chunks = retrieve_top_chunks(question, chunks)
        prompt = build_prompt(question, top_chunks)
        answer = ask_llm(prompt)

        print(f"\nAnswer:\n{answer}\n")
        print("Sources used:")
        for i, chunk in enumerate(top_chunks, start=1):
            print(f"  [Source {i}] {chunk['source']} (score: {chunk['score']:.3f})")


if __name__ == "__main__":
    main()
