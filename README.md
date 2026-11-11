# Kestrel Nova Policy Q&A Assistant (RAG with citations)

A REST service that answers employee questions using **only** the 10 Kestrel Nova policy
documents in `policies/`. Every answer cites the policy file, the section and an exact supporting
quote. When the documents don't contain the answer, the service says so instead of guessing.
It follows the project's API contract (`GET /health`, `POST /ask`).

## Quick start

Requires Python 3.10+ and internet access on first start (to download the embedding model, about 90 MB).

```bash
git clone https://github.com/davidkjeremiah/policy-qa-assistant.git && cd policy-qa-assistant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then set LLM_API_KEY (free Gemini key: https://aistudio.google.com/apikey)
python run.py                 # listens on port 8000 (override with PORT=...)
```

On Windows without `cp`, use `copy .env.example .env`. The key can also be supplied as an
environment variable (`LLM_API_KEY` or `GEMINI_API_KEY`) instead of a `.env` file.

Check it:
```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. Is that within the limit?"}'
```
```json
{
  "question": "I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. Is that within the limit?",
  "answered": true,
  "answer": "Yes. The domestic hotel limit is NGN 80,000 per night, so NGN 70,000 is within it.",
  "citations": [
    {"doc_id": "04_business_travel_policy", "section": "4",
     "quote": "Domestic hotels: maximum **NGN 80,000 per night**."}
  ]
}
```

Errors:
- A missing, empty, non-string or over 1000 character `question`, or a body that is not JSON, returns `422 {"error": "..."}`.
- A model provider failure (bad key, outage, timeout) or unusable model output returns `500 {"error": "..."}`.
- A failure never returns a `200` refusal.

## Models used

| Role | Model | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384 dimensions, run through `fastembed` / ONNX) | Free, fast on CPU, strong for short English text. ONNX avoids a PyTorch install and keeps memory use low. |
| Generation | `gemini-2.5-flash` through Gemini's OpenAI-compatible endpoint (configurable) | Free tier, answers in a few seconds (well inside the 60 s limit), reliable JSON output and instruction following. |

Any OpenAI-compatible provider works by changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`
in `.env` (examples for OpenAI, Groq and a local Ollama model are in `.env.example`).
A local 8B model on a CPU may exceed the 60 s limit.

## Design choices

**How documents are split.** There is one chunk per top-level section (`## N. Title`), 48 chunks across the 10 policies.
- *Why:* the sections are short (about 30 to 90 words) and self-contained, each covering one rule area.
- Splitting on them keeps each rule intact without size-based splitting or overlap.
- It also makes the citation `section` field exact.
- Each chunk's document title and section title are prepended before embedding, so a question about "per diem" matches the travel policy.
- The disclaimer, title and version lines above section 1 have no section number to cite, so they are not indexed.

**Embedding model and retrieval.** MiniLM embeddings sit in an in-memory NumPy matrix, alongside a BM25 keyword index. The two rankings are merged with reciprocal rank fusion.
- *Why:* 48 chunks need no vector database.
- Policy questions often hinge on exact terms ("per diem", "TOIL", "probation", "NGN 50,000") that dense embeddings blur and BM25 catches.
- BM25 tokens go through a tiny suffix stripper, so "approve", "approved" and "Approvals" match each other.

**How many chunks are retrieved.** `TOP_K = 8` sections, about 500 to 600 words of context.
- *Why:* this covers questions that need two documents. For example, a hotel claim involves both the travel policy (limits) and the expense policy (approvals).
- In development, the expense approvals section ranked 6th for the hotel question, so 8 leaves headroom for ranking errors while keeping the context small and focused.

**How citations are made exact.** Every sentence or bullet in the retrieved sections is labelled with an ID such as `[E2.3]`.
- The model cites IDs instead of writing quotes.
- The code copies the exact text of each cited sentence from the source file into `quote`.
- Quotes are therefore always verbatim, and only the sentences the answer relies on are cited.

**How refusal works.** There are three layers, and each one fails safe:
1. **Retrieval gate.** If the best cosine similarity is below `MIN_SCORE` (0.20), the
   question is clearly off-topic. The service refuses without calling the model.
2. **Model abstention.** The prompt forbids outside knowledge and requires JSON with an
   explicit `answerable` flag. `false` produces the standard refusal with `citations: []`.
3. **Citation validation.** An answer with no valid citation ID is converted into a refusal,
   so `answered: true` always has at least one real citation.

Generation runs at temperature 0. Infrastructure failures (provider down, invalid model output
after one retry) return HTTP 500 rather than a refusal, as the contract requires. All model calls
for one request share a 50 s budget (`REQUEST_BUDGET`), so a request never runs past 60 s.

## Evaluation

```bash
python run.py                    # in one terminal
python evaluate.py               # in another: writes results.csv
# fill in outcome + notes in results.csv (manual judgement against the policies)
python evaluate.py --summarize   # validates outcomes and updates the summary below
```

`evaluate.py` also checks automatically that every returned quote appears word for word in its
source file (ignoring `**`, `#`, `-` and extra spaces, as the contract does), and records any
mismatch in `notes`.

### Results summary
<!-- EVAL_SUMMARY_START -->
| Result | Count |
|---|---|
| Correct answers | 12 |
| Correct refusals | 3 |
| Failures | 0 |
| **Total** | **15** |

Failures by type: partially correct 0, incorrect 0, wrong refusal 0, missed refusal 0. Each failure is explained in the `notes` column of `results.csv`.
<!-- EVAL_SUMMARY_END -->

Run with `gemini-2.5-flash`, `TOP_K=8`, `MIN_SCORE=0.20`; about 5 s per question.

### Observations and near misses
- **No failures on the 15 questions, but the test set is small.** All 12 answerable questions were answered with the right document and section, and every quote passed the verbatim check.
- **All three refusals came from the model (layer 2), not the retrieval gate.** The off-topic questions (pension, dog, stock options) still scored 0.30 to 0.38 cosine against loosely related sections, above the 0.20 gate. The gate only catches clearly unrelated questions; correct refusal of near-topic questions relies on the prompt and the `answerable` flag.
- **Multi-document questions are the weakest point for retrieval.** For Q11 (hotel limit plus approver) the expense approvals section ranked 9th without BM25 stemming and only 6th with it. This is why `TOP_K` is 8, not 6.
- **Q11 involves an interpretation.** The answer applies the NGN 50,000 approval threshold to the per night amount and does not mention that hotels are normally booked through the Travel Desk. This is noted in `results.csv`.
- The first run produced stray spaces where the model had put citation IDs inside the answer text. Those IDs are now stripped before the answer is returned.

### What I would improve
- Build a larger test set (paraphrases, multi-part and near-topic questions) and track retrieval recall separately from answer quality.
- Follow cross-references between policies (for example travel section 6 pointing to the Expense Reimbursement Policy, FIN-01) so linked sections are always retrieved together.
- Add a cheap verification pass that checks each answer sentence against its cited quotes, to catch unsupported details.

## Known limitations
- Retrieval works at section level. A question that needs a section ranked below the top 8 will be wrongly refused or only partly answered.
- `MIN_SCORE` is calibrated for MiniLM and must be re-tuned if the embedding model changes.
- Answers depend on the hosted LLM. Rate limits or outages return 500 errors.
- There is no conversation memory, so each question is answered independently.
