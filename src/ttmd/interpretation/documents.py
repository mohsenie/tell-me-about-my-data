"""User documentation as a knowledge source for root-cause reasoning.

Extracts text from customer docs (PDF etc.), chunks it with page provenance,
and retrieves the most relevant passages for a given query (signals / topic).
Retrieved passages are fed to the LLM as CITED evidence, so hypotheses can be
grounded in the manufacturer's own manual.

Discipline: every passage carries its source (file + page). The interpreter must
cite what it used. No claim without a citation.

v1 retrieval is keyword/overlap based (no embeddings dependency). A vector-based
retriever can drop in behind the same `search()` interface later.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Chunk:
    doc: str
    page: int
    text: str

    def citation(self) -> str:
        return f"{self.doc} p.{self.page}"

    def to_dict(self) -> dict:
        return asdict(self)


_WORD = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _decamel(s: str) -> str:
    """Split CamelCase / snake / prefixed names into words so signal names like
    'EngineSpeed' or 'engine__EngineSpeed' match natural-language manual text."""
    s = s.replace("__", " ").replace("_", " ")
    s = _CAMEL.sub(" ", s)
    return s


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(_decamel(s).lower()))


def extract_pdf(path: str | Path, chunk_chars: int = 800) -> list[Chunk]:
    """Extract text from a PDF into page-tagged chunks."""
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        # split long pages into ~chunk_chars pieces on sentence-ish boundaries
        for piece in _split(text, chunk_chars):
            chunks.append(Chunk(doc=path.name, page=i, text=piece))
    return chunks


def _split(text: str, size: int) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(cur) + len(sent) > size and cur:
            out.append(cur.strip()); cur = ""
        cur += " " + sent
    if cur.strip():
        out.append(cur.strip())
    return out


_EXTRACT_SYSTEM = """You extract durable, DIAGNOSTICALLY-USEFUL facts about a
{asset_type} from its documentation, for later reasoning about WHY sensor signals
relate to each other.

ONLY extract facts that could help explain a relationship or anomaly between
sensor readings, such as:
- physical layout / proximity (e.g. "the fuel line runs near the turbocharger"),
- how a system works or what drives a reading (fuel, cooling, lubrication, boost),
- normal operating characteristics, limits, thermal/mechanical dependencies.

Do NOT extract (return nothing for these):
- document structure ("serial plate is on page 9", "views from port/starboard"),
- model/manufacturer identity, part numbers, legal/safety boilerplate,
- table-of-contents entries, section titles, generic warnings.

Rules:
- Only facts SUPPORTED by the text. Never invent.
- Each fact: one self-contained, mechanism-oriented sentence.
- If the excerpt contains nothing diagnostically useful, return an empty array.
Return ONLY a JSON array: [{{"fact": "..."}}] — no prose, no code fences."""

_EXTRACT_USER = """Documentation excerpts (from {asset_type} manual):
---
{text}
---
Extract only the diagnostically-useful {asset_type} facts as a JSON array."""

_BATCH_CHARS = 3500  # combine chunks up to this size per LLM call (fewer calls)


def extract_facts(chunks: list["Chunk"], asset_type: str, provider,
                  max_chunks: int = 40) -> list[dict]:
    """LLM pre-analysis: pull reusable, diagnostically-useful asset facts.

    Batches chunks per call (cost) and filters for diagnostic relevance (quality).
    Returns [{"fact": str, "citation": str}] — stored as a distinct, unverified
    tier in the knowledge base.
    """
    system = _EXTRACT_SYSTEM.format(asset_type=asset_type)
    usable = [c for c in chunks[:max_chunks] if len(c.text) >= 120]

    facts: list[dict] = []
    for batch in _batch(usable, _BATCH_CHARS):
        # citation spans the pages in the batch
        pages = sorted({c.page for c in batch})
        doc = batch[0].doc
        cite = (f"{doc} p.{pages[0]}" if len(pages) == 1
                else f"{doc} pp.{pages[0]}-{pages[-1]}")
        text = "\n\n".join(f"[p.{c.page}] {c.text}" for c in batch)
        try:
            raw = provider.complete(system, _EXTRACT_USER.format(
                text=text[:_BATCH_CHARS + 500], asset_type=asset_type)).strip()
            parsed = _parse_json_array(raw)
        except Exception:
            parsed = []
        for item in parsed:
            f = (item.get("fact") or "").strip()
            if f and not _looks_like_noise(f):
                facts.append({"fact": f, "citation": cite})
    return _dedupe(facts)


def _batch(chunks: list, max_chars: int):
    """Group consecutive chunks into batches under max_chars."""
    batch, size = [], 0
    for c in chunks:
        if size + len(c.text) > max_chars and batch:
            yield batch
            batch, size = [], 0
        batch.append(c); size += len(c.text)
    if batch:
        yield batch


_NOISE = re.compile(
    r"\b(page \d|table of contents|serial|part number|manufactured by|"
    r"port[- ]side|starboard[- ]side|documented with|section \d)\b", re.I)


def _looks_like_noise(fact: str) -> bool:
    """Belt-and-suspenders filter for non-diagnostic facts the prompt should
    already exclude (document structure, identity, etc.)."""
    return bool(_NOISE.search(fact))


def _parse_json_array(raw: str) -> list:
    import json, re
    # tolerate code fences / stray prose around the JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _dedupe(facts: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in facts:
        key = f["fact"].lower()
        if key not in seen:
            seen.add(key); out.append(f)
    return out


class DocumentIndex:
    """Keyword-overlap index over document chunks. Swap for vectors later."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._toks = [_tokens(c.text) for c in chunks]

    @classmethod
    def build(cls, doc_dir: str | Path) -> "DocumentIndex":
        doc_dir = Path(doc_dir)
        chunks: list[Chunk] = []
        for pdf in sorted(doc_dir.glob("*.pdf")):
            chunks.extend(extract_pdf(pdf))
        return cls(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for c, t in zip(self.chunks, self._toks):
            if not t:
                continue
            overlap = len(q & t)
            if overlap:
                score = overlap / (len(q) ** 0.5 * len(t) ** 0.5)  # cosine-ish
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"chunks": [c.to_dict() for c in self.chunks]}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DocumentIndex":
        data = json.loads(Path(path).read_text())
        return cls([Chunk(**c) for c in data["chunks"]])

    def __len__(self) -> int:
        return len(self.chunks)
