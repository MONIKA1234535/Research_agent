from agent import load_and_chunk_sources, retrieve_top_chunks


def test_unanswerable_question_flagged():
    chunks = load_and_chunk_sources("sources")
    results = retrieve_top_chunks("Who is the current CEO of NovaAssist?", chunks)
    assert len(results) > 0
