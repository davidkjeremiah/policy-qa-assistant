"""Evaluation for API_CONTRACT section 4.

  python evaluate.py               # run test_questions.csv through POST /ask and write results.csv
  # then fill in `outcome` (and `notes` for failures) in results.csv by hand
  python evaluate.py --summarize   # validate outcomes, print counts, update README
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import requests

from app.ingest import normalize

COLUMNS = ["id", "question", "answered", "answer", "citations", "outcome", "notes"]
OUTCOMES = {"correct", "partially_correct", "incorrect", "correct_refusal",
            "wrong_refusal", "missed_refusal"}
FAILURES = ("partially_correct", "incorrect", "wrong_refusal", "missed_refusal")


def col(row: dict, *names: str) -> str:
    norm = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
    return next((norm[n] for n in names if n in norm), "")


def run(args):
    with open(args.questions, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    sources = {p.stem: normalize(p.read_text(encoding="utf-8"))
               for p in Path(args.policies).glob("*.md")}

    out = []
    for i, row in enumerate(rows, 1):
        qid, q = col(row, "id") or str(i), col(row, "question")
        answered, answer, refs, notes = "", "", [], []
        try:
            r = requests.post(args.url, json={"question": q}, timeout=70)
            body = r.json()
            if r.status_code != 200:
                notes.append(f"HTTP {r.status_code}: {body.get('error', '')}")
            else:
                answered = "true" if body["answered"] else "false"
                answer = body["answer"]
                for c in body["citations"]:
                    ref = f"{c['doc_id']}#{c['section']}"
                    if ref not in refs:
                        refs.append(ref)
                    if normalize(c["quote"]) not in sources.get(c["doc_id"], ""):
                        notes.append(f"quote not verbatim in {c['doc_id']}")
                if body["answered"] and not body["citations"]:
                    notes.append("contract violation: answered without citations")
        except Exception as e:
            notes.append(f"request failed: {e}")

        out.append({"id": qid, "question": q, "answered": answered, "answer": answer,
                    "citations": ";".join(refs), "outcome": "", "notes": "; ".join(notes)})
        print(f"[{qid}] answered={answered or 'ERR':5s} {refs} {q[:60]}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {args.out}. Now fill in `outcome` (and `notes` for failures), "
          f"then run: python evaluate.py --summarize")


def summarize(args):
    with open(args.out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if r["outcome"].strip() not in OUTCOMES]
    if bad:
        sys.exit(f"Missing or invalid outcome for ids: {bad}. Allowed: {sorted(OUTCOMES)}")
    no_notes = [r["id"] for r in rows if r["outcome"] in FAILURES and not r["notes"].strip()]
    if no_notes:
        print(f"WARNING: failures without notes: {no_notes}")

    c = Counter(r["outcome"].strip() for r in rows)
    fails = sum(c[k] for k in FAILURES)
    summary = (
        "| Result | Count |\n|---|---|\n"
        f"| Correct answers | {c['correct']} |\n"
        f"| Correct refusals | {c['correct_refusal']} |\n"
        f"| Failures | {fails} |\n"
        f"| **Total** | **{len(rows)}** |\n\n"
        f"Failures by type: partially correct {c['partially_correct']}, incorrect {c['incorrect']}, "
        f"wrong refusal {c['wrong_refusal']}, missed refusal {c['missed_refusal']}. "
        "Each failure is explained in the `notes` column of `results.csv`."
    )
    print(summary)
    readme = Path("README.md")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        readme.write_text(re.sub(
            r"(<!-- EVAL_SUMMARY_START -->).*?(<!-- EVAL_SUMMARY_END -->)",
            lambda m: f"{m.group(1)}\n{summary}\n{m.group(2)}", text, flags=re.S), encoding="utf-8")
        print("README.md updated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="test_questions.csv")
    ap.add_argument("--policies", default="policies")
    ap.add_argument("--url", default="http://localhost:8000/ask")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
