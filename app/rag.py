import json
import logging
import re
import time
from dataclasses import dataclass, field

from . import config
from .ingest import load_sections
from .llm import chat
from .retriever import Retriever

log = logging.getLogger("policy_qa")

REFUSAL = "The policy documents do not contain information about this."

SYSTEM_PROMPT = """You are the internal policy assistant for Kestrel Nova Ltd.
Answer the employee's question using ONLY the policy excerpts in the user message.
Every sentence in the excerpts has an ID such as [E2.3].

Rules:
1. Use only facts stated in the excerpts. No outside knowledge, no assumptions, no guessing.
2. You may apply the rules to the employee's situation and do simple arithmetic
   (for example nights x per diem), provided every rule you use is stated in the excerpts.
3. If the excerpts do not contain the information needed to answer the question, set
   "answerable" to false. Never answer a different or more general question instead.
   If only part of the question is covered, answer that part and state clearly which part
   the policies do not cover.
4. Cite the IDs of the specific sentences your answer relies on, and only those.
   If the answer uses more than one document, cite each document used.
5. Answer in plain language in 1 to 4 sentences, keeping exact figures, deadlines and approvers.

Reply with JSON only, in exactly this form:
{"reasoning": "<one or two sentences on which excerpts are relevant>",
 "answerable": true or false,
 "answer": "<the answer, or empty if not answerable>",
 "citations": ["E1.2", "..."]}"""

ID_RE = re.compile(r"\[?\s*E\s*(\d+)\s*\.\s*(\d+)\s*\]?", re.I)   # lenient: reads cited IDs
STRIP_RE = re.compile(r"\[\s*E\d+\.\d+(?:\s*,\s*E\d+\.\d+)*\s*\]|\bE\d+\.\d+\b")   # strict: IDs in answer text


@dataclass
class Citation:
    doc_id: str
    section: str
    quote: str


@dataclass
class Result:
    answered: bool
    answer: str
    citations: list[Citation] = field(default_factory=list)
    reason: str = ""
    top_score: float = 0.0


class ModelOutputError(RuntimeError):
    pass


def parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw or "", re.S)
    for candidate in (raw, match.group(0) if match else None):
        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except ValueError:
                pass
    return {}


def truthy(v) -> bool:
    return v is True or str(v).strip().lower() in {"true", "yes"}


class PolicyRAG:
    def __init__(self):
        self.sections = load_sections(config.POLICY_DIR)
        if not self.sections:
            raise RuntimeError(f"No policy sections found in '{config.POLICY_DIR}'")
        self.retriever = Retriever(self.sections, config.EMBED_MODEL)
        log.info("Indexed %d sections", len(self.sections))

    def _refuse(self, reason: str, top: float) -> Result:
        log.info("refused: %s (top=%.3f)", reason, top)
        return Result(False, REFUSAL, [], reason, top)

    @staticmethod
    def _build_context(hits):
        ids, blocks = {}, []
        for e, (sec, _) in enumerate(hits, 1):
            lines = [f"### Excerpt E{e}: {sec.doc_title} (doc_id {sec.doc_id}), "
                     f"section {sec.number}. {sec.title}"]
            for s, unit in enumerate(sec.units, 1):
                sid = f"E{e}.{s}"
                ids[sid] = (sec, unit)
                lines.append(f"[{sid}] {unit}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), ids

    def _ask_model(self, question: str, context: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"},
        ]
        deadline = time.monotonic() + config.REQUEST_BUDGET   # stay inside the 60 s contract limit
        for _ in range(2):
            data = parse_json(chat(messages, deadline))
            if "answerable" in data:
                return data
            log.warning("model returned unusable output, retrying")
        raise ModelOutputError("model did not return valid JSON")

    def answer(self, question: str) -> Result:
        hits, top = self.retriever.search(question, config.TOP_K)

        # Layer 1: nothing relevant retrieved, so refuse without calling the model.
        if top < config.MIN_SCORE:
            return self._refuse("low_retrieval_score", top)

        context, ids = self._build_context(hits)
        data = self._ask_model(question, context)

        # Layer 2: the model says the excerpts don't answer the question.
        if not truthy(data.get("answerable")):
            return self._refuse("model_abstained", top)

        # Layer 3: keep only citation IDs that exist; quotes are copied from the source file.
        citations, seen = [], set()
        for raw in data.get("citations") or []:
            m = ID_RE.search(str(raw))
            sid = f"E{m.group(1)}.{m.group(2)}" if m else None
            if sid in ids and sid not in seen:
                seen.add(sid)
                sec, unit = ids[sid]
                citations.append(Citation(sec.doc_id, sec.number, unit))

        # Strip any citation IDs the model left in the answer text, and the gaps they leave.
        text = STRIP_RE.sub("", str(data.get("answer") or ""))
        text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\(\s*\)", "", text))
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text or not citations:
            return self._refuse("no_valid_citation", top)
        return Result(True, text, citations, "", top)
