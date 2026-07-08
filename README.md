Research Agent (with Citations)

What it does
Takes a natural-language question, retrieves the most relevant passages from a
set of local .txt source documents using TF-IDF + cosine similarity, and asks
an LLM to answer using only those retrieved passages, citing the source for
every claim. If the sources don't contain the answer, it says so explicitly
instead of guessing.

Tech stack
Python 3
scikit-learn (TfidfVectorizer, cosine_similarity) — retrieval
Groq API, model llama-3.3-70b-versatile — answer synthesis
python-dotenv — API key management

Setup
Clone this repo and cd into it
Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate

Install dependencies:
   pip install -r requirements.txt

Get a free API key at console.groq.com
Create a .env file in the project root:
   GROR_API_KEY=your_key_here

Run
   python agent.py

Type a question when prompted, or type quit to exit.

Source documents
Located in sources/:
*company_policy.txt — a company AI usage policy
*product_spec.txt — a product spec for a voice AI assistant
*research_paper.txt — a research summary on ML-based seizure detection
*ai_hiring.txt — a news report on AI hiring tools

Sample Q&A (from sample_run.txt)
Q: What AI tools are employees approved to use according to the policy?
According to the policy, employees are approved to use AI assistants including
Claude, ChatGPT, and Groq-hosted open models [Source 1].
Sources used: ai_policy.txt (score 0.327), ai_policy.txt (score 0.284), ai_policy.txt (score 0.197)

Q: What was the sensitivity of the hybrid EEG seizure detection model?
The sensitivity of the hybrid EEG seizure detection model was 91.2% [Source 1]
and [Source 3]. This is confirmed in both the discussion section [Source 1] and
the results section [Source 3].
Sources used: eeg_research_summary.txt (0.281, 0.221, 0.214)

Q: What is NovaAssist's pricing for the Enterprise tier exact dollar amount?
The provided sources do not contain enough information to answer this question.
Sources used: product_spec.txt (0.249, 0.099, 0.095)

Q: Who is the current CEO of NovaAssist?
The provided sources do not contain enough information to answer this question.

Full transcript of all 8 questions is in sample_run.txt.

Approach
Retrieval uses TF-IDF vectorization over paragraph-level chunks of all source
documents, with cosine similarity against the question vector to select the
top 3 most relevant chunks. This was chosen over embedding-based retrieval for
speed and zero external model dependency within a 24-hour build window — TF-IDF
runs instantly with no download or GPU requirement.
The retrieved chunks are inserted into a prompt that explicitly restricts the
LLM to only the numbered sources provided, requires inline citation of the
source number for every claim, and specifies an exact fallback sentence when
the answer isn't present. Temperature is set to 0.2 to keep answers factual and
low-variance rather than creative.

Tradeoffs and limitations
*TF-IDF vs. embeddings: TF-IDF matches on word overlap, so it can miss
semantically related passages that use different wording (a synonym-heavy
question could under-retrieve). Sentence embeddings would catch this but add model download time and latency — not worth it for this scope.
*Paragraph-level chunking: coarse compared to sentence-level; citations
point to the right document reliably but not always the exact sentence.
*No re-ranking step: a cross-encoder re-ranker after TF-IDF would improve
precision on ambiguous questions but was out of scope for 24 hours.
*Exit command: the loop only recognizes quit to exit, not exit — a
known minor gap; typing exit is treated as a literal question instead of a
command.
*Correct "no answer" behavior verified: two questions were deliberately
designed to be unanswerable from the sources (Enterprise pricing, CEO name),and both correctly triggered the fallback message rather than a hallucinated answer.