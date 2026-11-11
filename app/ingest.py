"""Parse policy Markdown into top-level sections and quotable sentence units."""
import re
from dataclasses import dataclass
from pathlib import Path

TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")               # "# Annual Leave Policy"
SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$")  # "## 4. Accommodation"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(])")


@dataclass
class Section:
    doc_id: str        # file name without .md
    doc_title: str
    number: str        # top-level section number, e.g. "4"
    title: str
    text: str          # raw Markdown body of the section
    units: list[str]   # sentences and bullets, verbatim from the file (used as quotes)


def normalize(s: str) -> str:
    """Normalisation used by the contract's quote check: ignore **, #, - and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[*#-]", " ", s)).strip()


def quotable_units(body: str) -> list[str]:
    units = []
    for line in body.splitlines():
        line = re.sub(r"^[-*]\s+", "", line.strip())    # drop the bullet marker only
        if line:
            units.extend(u.strip() for u in SENT_SPLIT.split(line) if u.strip())
    return units


def load_sections(policy_dir: str) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        doc_title, heads = path.stem, []
        for i, line in enumerate(lines):
            if doc_title == path.stem and (m := TITLE_RE.match(line)):
                doc_title = m.group(1)
            if m := SECTION_RE.match(line):
                heads.append((i, m.group(1), m.group(2)))
        for j, (i, num, title) in enumerate(heads):
            end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
            body = "\n".join(lines[i + 1:end]).strip()
            if body:
                sections.append(Section(path.stem, doc_title, num, title, body, quotable_units(body)))
    return sections
