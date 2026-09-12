"""Functional tests for the actors layer: registry CRUD + resolve-with-clarify,
and the ChatDeps actor capability methods. Pure/deterministic, no LLM, no vessel
data. Uses a temp store so artifacts/actors.json is never touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from galene.interpretation.actors import ActorRegistry, AmbiguousActor


# ---------------- registry: CRUD ----------------
def test_add_and_get(tmp_path):
    r = ActorRegistry(tmp_path / "actors.json")
    a = r.add("Andrew", description="engine room technician",
              contact_label="andrew@ship")
    assert a["name"] == "Andrew" and a["kind"] == "person"
    assert a["description"] == "engine room technician"
    assert a["contact"]["label"] == "andrew@ship"
    # forward-compat seams exist and start empty
    assert a["roles"] == [] and a["relationships"] == []
    assert r.get(a["id"]) is not None


def test_add_is_idempotent_by_name(tmp_path):
    r = ActorRegistry(tmp_path / "actors.json")
    r.add("Andrew", description="technician")
    r.add("Andrew", description="chief technician")   # same kind+name -> update
    assert len(r.all()) == 1
    assert r.all()[0]["description"] == "chief technician"


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "actors.json"
    ActorRegistry(p).add("Maria", description="captain")
    reloaded = ActorRegistry(p)          # fresh instance reads from disk
    assert len(reloaded.all()) == 1 and reloaded.all()[0]["name"] == "Maria"


def test_delete(tmp_path):
    r = ActorRegistry(tmp_path / "actors.json")
    a = r.add("Andrew")
    assert r.delete(a["id"]) is True
    assert r.get(a["id"]) is None
    assert r.delete("nonexistent") is False


# ---------------- registry: resolve-with-clarify ----------------
def test_resolve_single_and_unknown(tmp_path):
    r = ActorRegistry(tmp_path / "actors.json")
    r.add("Maria", description="captain")
    assert r.resolve("Maria")["description"] == "captain"
    assert r.resolve("maria")["name"] == "Maria"     # case-insensitive
    assert r.resolve("Bob") is None                  # unknown -> None, not error


def test_resolve_ambiguous_raises_with_candidates(tmp_path):
    r = ActorRegistry(tmp_path / "actors.json")
    r.add("Andrew", description="engine room technician")
    # a genuine second same-name actor (distinct id) forces ambiguity
    r._actors.append({"id": "z2", "kind": "person", "name": "Andrew",
                      "description": "captain", "contact": {}, "roles": [],
                      "relationships": []})
    with pytest.raises(AmbiguousActor) as exc:
        r.resolve("Andrew")
    assert len(exc.value.candidates) == 2
    descs = {c["description"] for c in exc.value.candidates}
    assert descs == {"engine room technician", "captain"}


# ---------------- ChatDeps actor methods ----------------
@pytest.fixture
def deps_with_temp_actors(deps, tmp_path):
    """ChatDeps wired to a temp actor store (so artifacts/ is untouched)."""
    deps._actors = ActorRegistry(tmp_path / "actors.json")
    return deps


def test_chatdeps_add_list_delete(deps_with_temp_actors):
    d = deps_with_temp_actors
    assert "No actors" in d.list_actors()
    out = d.add_actor("Andrew", "engine room technician", "andrew@ship")
    assert "Andrew" in out and "technician" in out
    listing = d.list_actors()
    assert "Andrew" in listing and "technician" in listing
    assert "Deleted" in d.delete_actor("Andrew")
    assert "No actors" in d.list_actors()


def test_chatdeps_delete_unknown_is_graceful(deps_with_temp_actors):
    assert "don't have an actor" in deps_with_temp_actors.delete_actor("Nobody")


def test_chatdeps_resolve_ambiguous_asks(deps_with_temp_actors):
    from galene.interpretation.orchestrator import _NeedsClarification
    d = deps_with_temp_actors
    d.add_actor("Andrew", "technician")
    d.actor_registry()._actors.append(
        {"id": "z", "kind": "person", "name": "Andrew", "description": "captain",
         "contact": {}, "roles": [], "relationships": []})
    with pytest.raises(_NeedsClarification):
        d.resolve_actor("Andrew")
