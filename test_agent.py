from pathlib import Path

from agent import load_and_chunk_sources, retrieve_top_chunks

SOURCES_DIR = Path(__file__).resolve().parent / "sources"


def test_load_and_chunk_sources_returns_non_empty_list():
    chunks = load_and_chunk_sources(SOURCES_DIR)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert chunk["text"].strip()
        assert chunk["source"].endswith(".txt")


def test_retrieve_top_chunks_returns_top_k_sorted_by_score():
    chunks = load_and_chunk_sources(SOURCES_DIR)
    question = "What AI tools are employees approved to use according to the policy?"
    top_k = 3

    results = retrieve_top_chunks(question, chunks, top_k=top_k)

    assert len(results) == top_k
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert "text" in result
        assert "source" in result
        assert "score" in result
        assert isinstance(result["score"], float)


def test_unanswerable_question_still_returns_chunks():
    chunks = load_and_chunk_sources(SOURCES_DIR)
    question = "Who is the current CEO of NovaAssist?"

    results = retrieve_top_chunks(question, chunks, top_k=3)

    assert len(results) == 3
    for result in results:
        assert result["text"]
        assert result["source"]
        assert "score" in result
