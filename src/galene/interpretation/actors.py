"""Actors layer — a small, user-editable directory of WHO (people now; roles +
org relationships in a later iteration) so a watch/notification can name a person
("notify Andrew") and the system resolves it, asking a clarifying question when
the name is ambiguous.

Design mirrors the knowledge base: attributed, dated, editable, stored as JSON.
The schema is deliberately FORWARD-COMPATIBLE so roles + auth slot in without a
migration:
  - `kind`          : "person" today; "role" reserved for the roles iteration.
  - `roles`         : list of role ids a person holds (empty until roles exist).
  - `relationships` : list of {type, target_id} (e.g. reports-to) — empty for now.
  - `contact`       : {label, ...} — a LABEL only today; actual delivery
                      (email/webhook) is a separate opt-in step, not stored as a
                      live channel here.

NO authentication/authorization: an "admin" concept is NOT enforced anywhere yet
(explicitly out of scope — see TODO). This module only stores and resolves.

Stored as JSON at artifacts/actors.json (global — people aren't vessel-scoped).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _actor_id(name: str, kind: str = "person") -> str:
    """Stable short id from kind+name, so references survive list reordering."""
    return hashlib.sha1(f"{kind}:{name.strip().lower()}".encode()).hexdigest()[:8]


class AmbiguousActor(Exception):
    """Raised by resolve() when a name matches more than one actor. Carries the
    candidate actors so the caller can ask a clarifying question. Kept provider-
    free (no orchestrator import) — the chat layer maps this to its clarification
    flow."""

    def __init__(self, name: str, candidates: list[dict]):
        self.name = name
        self.candidates = candidates
        super().__init__(f"'{name}' matches {len(candidates)} actors")


class ActorRegistry:
    """CRUD + resolution over the actors store. Pure data + JSON persistence."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._actors: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._actors = data.get("actors", [])

    # --- read ---
    def all(self) -> list[dict]:
        return list(self._actors)

    def get(self, actor_id: str) -> dict | None:
        return next((a for a in self._actors if a["id"] == actor_id), None)

    def _by_name(self, name: str) -> list[dict]:
        low = (name or "").strip().lower()
        if not low:
            return []
        # exact (case-insensitive) name match; a later iteration can add role-name
        # matching ("the captain") once roles exist.
        return [a for a in self._actors if a.get("name", "").strip().lower() == low]

    # --- create / update / delete ---
    def add(self, name: str, kind: str = "person", description: str = "",
            contact_label: str | None = None, author: str = "user") -> dict:
        """Add an actor. If one with the same kind+name exists, update its fields
        instead of duplicating (idempotent by kind+name)."""
        name = (name or "").strip()
        if not name:
            raise ValueError("an actor needs a name")
        aid = _actor_id(name, kind)
        existing = self.get(aid)
        if existing:
            if description:
                existing["description"] = description
            if contact_label is not None:
                existing.setdefault("contact", {})["label"] = contact_label
            existing["updated_at"] = _now()
            self.save()
            return existing
        actor = {
            "id": aid,
            "kind": kind,                       # "person" now; "role" later
            "name": name,
            "description": description,          # e.g. "engine room technician"
            "contact": {"label": contact_label} if contact_label else {},
            "roles": [],                         # seam: role ids (roles iteration)
            "relationships": [],                 # seam: [{type, target_id}]
            "author": author,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._actors.append(actor)
        self.save()
        return actor

    def update(self, actor_id: str, **fields) -> dict | None:
        a = self.get(actor_id)
        if not a:
            return None
        for k in ("name", "description", "kind"):
            if k in fields and fields[k] is not None:
                a[k] = fields[k]
        if fields.get("contact_label") is not None:
            a.setdefault("contact", {})["label"] = fields["contact_label"]
        a["updated_at"] = _now()
        self.save()
        return a

    def delete(self, actor_id: str) -> bool:
        before = len(self._actors)
        self._actors = [a for a in self._actors if a["id"] != actor_id]
        if len(self._actors) != before:
            self.save()
            return True
        return False

    # --- resolution (the "who is Andrew?" step) ---
    def resolve(self, name: str) -> dict | None:
        """Resolve a name to exactly ONE actor.
          - 0 matches -> None (caller says "I don't know an <name>").
          - 1 match   -> that actor.
          - >1 match  -> raise AmbiguousActor(name, candidates) so the caller can
                         ask which one (e.g. two Andrews with different roles).
        """
        matches = self._by_name(name)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        raise AmbiguousActor(name, matches)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "actors": self._actors,
            "updated_at": _now(),
        }, indent=2))
