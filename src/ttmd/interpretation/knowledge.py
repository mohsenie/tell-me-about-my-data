"""Knowledge base of confirmed explanations AND general asset facts.

Two kinds of learned knowledge, per asset type:
  1. relationships: explanation tied to a specific signal pair (exact override).
  2. asset_facts:   GENERAL truths about the asset (e.g. "fuel line runs near the
     turbocharger; fuel gains heat under load"). These GENERALIZE — they are fed
     as context to ALL future interpretations so reasoning improves across
     seemingly unrelated relationships.

Only EXPERT-CONFIRMED knowledge is stored (never the LLM's own guesses), so the
system never learns from its own hallucinations. Everything is attributed,
dated, and editable. Retrieval-augmented, not fine-tuning.

Stored as JSON at artifacts/knowledge_<asset_type>.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _key(a: str, b: str) -> str:
    return "::".join(sorted([a, b]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fact_id(fact: str) -> str:
    """Stable short id from the fact text, so review survives list reordering."""
    return hashlib.sha1(fact.strip().lower().encode()).hexdigest()[:8]


class KnowledgeBase:
    def __init__(self, asset_type: str, path: Path):
        self.asset_type = asset_type
        self.path = Path(path)
        self._rel: dict = {}
        self._facts: list = []
        self._field_sem: dict = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._rel = data.get("relationships", {})
            self._facts = data.get("asset_facts", [])
            self._field_sem = data.get("field_semantics", {})

    # --- relationship-specific explanations (exact override) ---
    def confirmed(self, a: str, b: str) -> dict | None:
        return self._rel.get(_key(a, b))

    def add_correction(self, a: str, b: str, explanation: str,
                       author: str = "user", general_fact: str | None = None) -> None:
        self._rel[_key(a, b)] = {
            "signals": [a, b], "explanation": explanation,
            "source": "user_confirmed", "author": author, "updated_at": _now(),
        }
        # A correction usually implies a GENERAL asset fact that generalizes.
        if general_fact:
            self.add_asset_fact(general_fact, author=author, derived_from=[a, b])
        self.save()

    # --- general asset facts (generalize across relationships) ---
    def add_asset_fact(self, fact: str, author: str = "user",
                       derived_from: list | None = None,
                       source: str = "user_confirmed",
                       citation: str | None = None) -> None:
        entry = {
            "id": _fact_id(fact),
            "fact": fact, "author": author, "source": source,
            "derived_from": derived_from or [], "updated_at": _now(),
        }
        if citation:
            entry["citation"] = citation
        self._facts.append(entry)
        self.save()

    def add_document_facts(self, facts: list[dict], doc_source: str) -> int:
        """Add facts extracted from a document as a DISTINCT, lower tier.

        Each fact: {"fact": str, "citation": str}. Tagged source='document_extracted'
        (fallible extraction, not expert-confirmed). Replaces prior facts from the
        same document so re-running is idempotent.
        """
        self._facts = [f for f in self._facts
                       if not (f.get("source") == "document_extracted"
                               and f.get("doc") == doc_source)]
        for f in facts:
            self._facts.append({
                "id": _fact_id(f["fact"]),
                "fact": f["fact"],
                "source": "document_extracted",
                "doc": doc_source,
                "citation": f.get("citation", doc_source),
                "confidence": "unverified_extraction",
                "updated_at": _now(),
            })
        self.save()
        return len(facts)

    def asset_facts(self) -> list[dict]:
        return list(self._facts)

    # --- field semantics (name-based meaning proposals; lowest-confidence tier) ---
    def set_field_semantics(self, source: str, proposals: list[dict]) -> int:
        """Store proposed field meanings for a source (replaces prior for that
        source, so re-running is idempotent). Unverified until promoted."""
        self._field_sem = getattr(self, "_field_sem", {})
        for p in proposals:
            p = {**p, "source": source, "status": "proposed", "updated_at": _now()}
            self._field_sem[f"{source}::{p['field']}"] = p
        self.save()
        return len(proposals)

    def field_semantics(self, source: str | None = None) -> list[dict]:
        fs = getattr(self, "_field_sem", {})
        vals = list(fs.values())
        return [v for v in vals if source is None or v.get("source") == source]

    def confirmed_field(self, source: str, field: str) -> dict | None:
        fs = getattr(self, "_field_sem", {})
        e = fs.get(f"{source}::{field}")
        return e if e and e.get("status") == "confirmed" else None

    def promote_field(self, source: str, field: str, author: str = "user") -> dict:
        fs = getattr(self, "_field_sem", {})
        e = fs.get(f"{source}::{field}")
        if e is None:
            raise KeyError(f"no field semantics for {source}::{field}")
        e["status"] = "confirmed"; e["author"] = author; e["updated_at"] = _now()
        self.save()
        return e

    def extracted_facts(self) -> list[dict]:
        """Document-extracted (unverified) facts awaiting expert review."""
        return [f for f in self._facts if f.get("source") == "document_extracted"]

    def _find_extracted(self, fact_id: str) -> dict | None:
        for f in self._facts:
            if f.get("source") == "document_extracted" and f.get("id") == fact_id:
                return f
        return None

    def promote_fact(self, fact_id: str, author: str = "user") -> dict:
        """Promote an extracted fact (by STABLE id) to EXPERT-CONFIRMED.

        Id-based so review is unaffected by list reordering between actions.
        """
        target = self._find_extracted(fact_id)
        if target is None:
            raise KeyError(f"no extracted fact with id '{fact_id}'")
        target["source"] = "user_confirmed"
        target["author"] = author
        target["promoted_from"] = "document_extracted"
        target["updated_at"] = _now()
        self.save()
        return target

    def reject_fact(self, fact_id: str) -> dict:
        """Remove an extracted fact (by STABLE id) the expert judged wrong."""
        target = self._find_extracted(fact_id)
        if target is None:
            raise KeyError(f"no extracted fact with id '{fact_id}'")
        self._facts.remove(target)
        self.save()
        return target

    # --- prompt context (the generalization mechanism) ---
    def context_for_prompt(self, limit: int = 50) -> str:
        lines = []
        confirmed = [f for f in self._facts if f.get("source") == "user_confirmed"]
        extracted = [f for f in self._facts if f.get("source") == "document_extracted"]

        if confirmed:
            lines.append("EXPERT-CONFIRMED facts about this asset (authoritative):")
            for f in confirmed[:limit]:
                lines.append(f"  - {f['fact']}")
        if self._rel:
            lines.append("EXPERT-CONFIRMED relationship explanations:")
            for e in list(self._rel.values())[:limit]:
                lines.append(f"  - {', '.join(e['signals'])}: {e['explanation']}")
        if extracted:
            lines.append("FROM DOCUMENTATION (extracted; unverified — use with care, "
                         "prefer expert-confirmed facts if they conflict):")
            for f in extracted[:limit]:
                lines.append(f"  - {f['fact']} [{f.get('citation','doc')}]")
        return "\n".join(lines) if lines else "(no confirmed domain knowledge yet)"

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "asset_type": self.asset_type,
            "asset_facts": self._facts,
            "relationships": self._rel,
            "field_semantics": getattr(self, "_field_sem", {}),
        }, indent=2))

    def __len__(self) -> int:
        return len(self._rel) + len(self._facts)
