"""Command-line entry point — Relational Fingerprinting discovery.

Usage:
    ttmd discover <source>            # regimes + per-regime relationship graphs
    ttmd discover engine --no-mi      # skip mutual information (faster)

Reads partitioned ship data from ship-data/<vessel>/<source>/logs/date=*/*.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running from the repo root without install (adds ./config on path).
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402  (root-level project paths)
from ttmd import term  # noqa: E402
from ttmd.discovery.loader import load_numeric  # noqa: E402
from ttmd.discovery.relationships import build_per_regime_graphs  # noqa: E402
from ttmd.discovery.fusion import fuse_sources, is_cross_source  # noqa: E402
from ttmd.reporting import describe_discovery, render_markdown  # noqa: E402
from ttmd.interpretation import (  # noqa: E402
    get_provider, KnowledgeBase, interpret_report, apply_correction,
    check_scope, REFUSAL)
from ttmd.query import (  # noqa: E402
    list_capabilities, describe_capabilities, aggregate, fuel_consumption)
from ttmd.query.plots import scatter, timeseries, aggregated_series  # noqa: E402
from ttmd.interpretation.documents import DocumentIndex, extract_facts  # noqa: E402

from ttmd.cli_helpers import (  # noqa: E402
    kb as _kb, resolve_window as _resolve_window, present_globs,
    print_discovery_summary as _print_discovery_summary)


def _asset_type(args):
    """Asset type: explicit --asset-type flag, else declared in sources.yaml,
    else neutral fallback. Never hardcoded to 'ship'."""
    if getattr(args, "asset_type", None):
        return args.asset_type
    return config.asset_type(getattr(args, "vessel", config.DEFAULT_VESSEL))


def cmd_discover(args) -> None:
    globs, label = _resolve_window(args.source, args.vessel, args.days,
                                   getattr(args, "from"), args.to)
    if not globs:
        sys.exit(f"no data for source '{args.source}' in the requested window.")
    present = [g for g in globs if glob.glob(g)]
    if not present:
        sys.exit(f"no parquet found for '{args.source}' in window {label}.")

    print(f"window: {label}  ({len(present)} date partition(s))")
    cols, data = load_numeric(present)
    result = build_per_regime_graphs(cols, data, with_mi=not args.no_mi)
    result["window"] = label

    out = config.ARTIFACTS_DIR / f"discovery_{args.source}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    _print_discovery_summary(args.source, result)
    print(f"\nWrote {out.relative_to(config.ROOT)}")


def cmd_fused(args) -> None:
    # Auto-detect sources from the filesystem unless the user names specific ones.
    sources = args.sources or config.discover_sources(args.vessel)
    if len(sources) < 2:
        sys.exit(f"fusion needs >=2 sources; found: {sources or 'none'}")
    globs = {s: config.source_glob(s, args.vessel) for s in sources}
    cols, data = fuse_sources(globs)
    print(f"fused {len(sources)} sources {sources} -> {len(cols)} signals, "
          f"{len(data):,} aligned rows")

    result = build_per_regime_graphs(cols, data, with_mi=not args.no_mi)
    out = config.ARTIFACTS_DIR / "discovery_fused.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    _print_discovery_summary("fused", result)

    # honest regime-quality diagnostic: fusing heterogeneous sources blurs the
    # clustering, so warn if the fused regimes are low-separation (the cross-source
    # RELATIONSHIPS are the trustworthy output, not the fused modes).
    reg = result.get("regimes", {})
    sil = reg.get("silhouette")
    from ttmd.discovery.fusion import FUSED_SILHOUETTE_MIN
    if sil is not None and sil < FUSED_SILHOUETTE_MIN:
        print(f"\nNOTE: fused regimes are blurred (silhouette {sil:.2f} < "
              f"{FUSED_SILHOUETTE_MIN}) — fusing dissimilar sources lowers regime "
              "separation. Trust per-source operating modes; use fusion for the "
              "CROSS-SOURCE relationships below.")

    g = result["graphs"].get("global", {})
    cross = [e for e in g.get("edges", []) if is_cross_source(e)]
    cross.sort(key=lambda e: e["dcor"], reverse=True)
    print(f"\ncross-source relationships: {len(cross)} of {len(g.get('edges', []))} edges")
    for e in cross[:12]:
        print(f"  {e['a']:26s} ~ {e['b']:26s} r={e['pearson']:.2f} dcor={e['dcor']:.2f} [{e['kind']}]")
    print(f"\nWrote {out.relative_to(config.ROOT)}")


def cmd_baseline(args) -> None:
    """Set a KNOWN-GOOD baseline fingerprint for a source over a window.

    ttmd baseline engine --from 2026-09-01 --to 2026-09-03
    The window should be a period the operator considers healthy/normal.
    """
    from ttmd.anomaly import build_baseline, save_baseline
    globs, label = _resolve_window(args.source, args.vessel, args.days,
                                   getattr(args, "from"), args.to)
    dates = _selected_dates(args.source, args.vessel, args.days,
                            getattr(args, "from"), args.to)
    if not dates:
        sys.exit(f"no data for '{args.source}' in the requested window.")
    print(f"building known-good baseline for '{args.source}' over {label} "
          f"({len(dates)} day(s))...")
    bl = build_baseline(args.source, dates, args.vessel, with_mi=not args.no_mi)
    if not bl:
        sys.exit(f"could not build a baseline for '{args.source}'.")
    path = save_baseline(bl)
    reg = bl.get("regime_summary", {})
    band = bl.get("variation", {}).get("summary", {})
    print(f"  regimes: {reg.get('k', 0)}  "
          f"normal edge-variation band (p90 stddev): {band.get('p90_edge_std')}")
    print(f"Wrote {path.relative_to(config.ROOT)}")


def cmd_detect(args) -> None:
    """Compare a window to the known-good baseline and report drift."""
    from ttmd.anomaly import load_baseline, has_baseline, detect_drift
    from ttmd.anomaly.report import render_drift
    if not has_baseline(args.source):
        sys.exit(f"no baseline for '{args.source}'. Set one first: "
                 f"ttmd baseline {args.source} --from <date> --to <date>")
    dates = _selected_dates(args.source, args.vessel, args.days,
                            getattr(args, "from"), args.to)
    if not dates:
        sys.exit(f"no data for '{args.source}' in the requested window.")
    baseline = load_baseline(args.source)
    result = detect_drift(args.source, dates, args.vessel, baseline,
                          with_mi=not args.no_mi, use_ae=getattr(args, "ae", False))
    print(render_drift(result))


def _selected_dates(source, vessel, days, date_from, date_to):
    """Resolve a window to the list of date strings it covers (for baseline/detect)."""
    available = config.available_dates(source, vessel)
    if not available:
        return []
    if days:
        return available[-days:]
    if date_from or date_to:
        lo = date_from or available[0]
        hi = date_to or available[-1]
        return [d for d in available if lo <= d <= hi]
    return available


def cmd_report(args) -> None:
    """Human-readable data-intelligence report from a discovery artifact."""
    art = config.ARTIFACTS_DIR / f"discovery_{args.source}.json"
    if not art.exists():
        sys.exit(f"no discovery artifact for '{args.source}'. Run `ttmd discover {args.source}` first.")
    artifact = json.loads(art.read_text())
    report = describe_discovery(artifact, args.source)
    md = render_markdown(report)
    out = config.ARTIFACTS_DIR / f"report_{args.source}.md"
    out.write_text(md)
    print(md)
    print(f"\n---\nWrote {out.relative_to(config.ROOT)}")


def cmd_interpret(args) -> None:
    """Report + LLM root-cause hypotheses (asset-aware), preferring confirmed KB."""
    art = config.ARTIFACTS_DIR / f"discovery_{args.source}.json"
    if not art.exists():
        sys.exit(f"no discovery artifact for '{args.source}'. Run `ttmd discover {args.source}` first.")
    report = describe_discovery(json.loads(art.read_text()), args.source)
    kb = _kb(_asset_type(args))
    provider = get_provider()

    docs = None
    if not args.no_docs and config.USER_DOC_DIR.is_dir() and any(config.USER_DOC_DIR.glob("*.pdf")):
        docs = DocumentIndex.build(config.USER_DOC_DIR)

    def _progress(done, total):
        print(term.system(f"\r  reasoning... {done}/{total} LLM calls done"),
              end="", flush=True)

    print(term.system(f"interpreting (provider={type(provider).__name__})..."))
    report = interpret_report(report, _asset_type(args), provider, kb, docs=docs,
                              progress=_progress if provider.__class__.__name__ != "StubProvider" else None)
    print()  # newline after progress
    md = render_markdown(report)
    out = config.ARTIFACTS_DIR / f"report_{args.source}.md"
    out.write_text(md)
    print(md)
    print(f"\n---\nprovider={type(provider).__name__}  knowledge_entries={len(kb)}"
          f"  doc_chunks={len(docs) if docs else 0}")
    print(f"Wrote {out.relative_to(config.ROOT)}")


def cmd_ingest_docs(args) -> None:
    """Pre-analyze user documentation: extract durable asset facts into the KB.

    Extracted facts are stored as a DISTINCT, unverified tier (source=document),
    combined with expert-confirmed knowledge to improve reasoning. Needs an LLM
    provider (set TTMD_LLM=bedrock) for real extraction.
    """
    if not (config.USER_DOC_DIR.is_dir() and any(config.USER_DOC_DIR.glob("*.pdf"))):
        sys.exit(f"no PDFs in {config.USER_DOC_DIR}")
    provider = get_provider()
    kb = _kb(_asset_type(args))
    index = DocumentIndex.build(config.USER_DOC_DIR)
    print(f"indexed {len(index)} chunks; extracting facts (provider={type(provider).__name__})...")

    # group chunks by document so facts are attributed per file
    by_doc: dict[str, list] = {}
    for c in index.chunks:
        by_doc.setdefault(c.doc, []).append(c)

    total = 0
    for doc, chunks in by_doc.items():
        # Sample chunks EVENLY across the document (not just the first N), so a
        # cap doesn't miss the technical middle behind front-matter.
        if args.max_chunks and len(chunks) > args.max_chunks:
            step = len(chunks) / args.max_chunks
            chunks = [chunks[int(i * step)] for i in range(args.max_chunks)]
        facts = extract_facts(chunks, _asset_type(args), provider, max_chunks=len(chunks))
        n = kb.add_document_facts(facts, doc_source=doc)
        total += n
        print(f"  {doc}: extracted {n} fact(s)")
        for f in facts[:5]:
            print(f"     - {f['fact']}  [{f['citation']}]")
    print(f"\nadded {total} document fact(s). KB now has {len(kb)} item(s).")


def cmd_review_docs(args) -> None:
    """Expert review of document-extracted facts: list / promote / reject.

    Promote = the expert vouches for it (becomes authoritative). Reject = remove.
    """
    kb = _kb(_asset_type(args))
    if args.action == "list":
        # show the whole knowledge base by tier so the expert sees what's already
        # authoritative alongside what still needs review.
        confirmed = [f for f in kb.asset_facts()
                     if f.get("source") != "document_extracted"]
        ext = kb.extracted_facts()
        if confirmed:
            print(f"Expert-confirmed ({len(confirmed)}, authoritative):")
            for f in confirmed:
                print(f"  [{f['id']}] {f['fact']}")
            print()
        if not ext:
            print("No document-extracted facts awaiting review. "
                  "Run `ttmd ingest-docs` to extract from manuals.")
            return
        print(f"{len(ext)} extracted fact(s) awaiting review "
              "(promote = vouch/authoritative, reject = remove):\n")
        for f in ext:
            print(f"  [{f['id']}] {f['fact']}  [{f.get('citation','doc')}]")
        print("\nPromote: ttmd review-docs promote <id>")
        print("Reject:  ttmd review-docs reject <id>")
    elif args.action in ("promote", "reject"):
        if not args.fact_id:
            sys.exit(f"{args.action} needs a fact id. Run `ttmd review-docs list` first.")
        try:
            fact = (kb.promote_fact(args.fact_id) if args.action == "promote"
                    else kb.reject_fact(args.fact_id))
        except KeyError as e:
            sys.exit(str(e))
        verb = "Promoted to CONFIRMED" if args.action == "promote" else "Rejected"
        print(f"{verb}: {fact['fact']}")


def cmd_describe_fields(args) -> None:
    """LLM proposes field meanings, using USER-declared protocol context.

    Proposals are stored as an unverified 'field_semantics' tier for expert
    review (promote). Protocol context comes from ship-data/<vessel>/sources.yaml.
    """
    from ttmd.query.capabilities import list_capabilities
    from ttmd.interpretation.fields import describe_fields

    globs = {args.source: config.source_glob(args.source, args.vessel)}
    caps = list_capabilities(globs)
    if args.source not in caps:
        sys.exit(f"no data for source '{args.source}'.")
    cols = caps[args.source]["signals"]  # only queryable (varying) fields

    meta = config.source_metadata(args.vessel).get(args.source, {})
    protocol = meta.get("protocol")
    desc = meta.get("description")
    provider = get_provider()
    kb = _kb(_asset_type(args))

    print(term.system(f"describing {len(cols)} fields of '{args.source}' "
                      f"(protocol: {protocol or 'none declared'}, "
                      f"provider: {type(provider).__name__})..."))
    proposals = describe_fields(args.source, cols, _asset_type(args), provider,
                                protocol=protocol, source_description=desc)
    kb.set_field_semantics(args.source, proposals)

    print(term.heading(f"\nProposed field meanings for '{args.source}' "
                       "(unverified — review + promote):\n"))
    for p in proposals:
        std = "standard" if p["is_standard"] else "custom"
        conf = {"high": term.info, "medium": term.system, "low": term.warn}.get(
            p["confidence"], term.system)(p["confidence"])
        print(f"  {term.assistant(p['field'])}: {p['description']}")
        print(term.system(f"      unit={p['unit'] or '?'}  agg={p['aggregation']}  "
                          f"{std}  confidence=") + conf)
    print(term.system(f"\nStored {len(proposals)} proposals. Confirm with your "
                      "review step; queries can then use confirmed units/aggregations."))


def cmd_capabilities(args) -> None:
    """List what the user can query (available signals per source)."""
    sources = ([args.source] if args.source
               else config.discover_sources(args.vessel))
    globs = {s: config.source_glob(s, args.vessel) for s in sources}
    caps = list_capabilities(globs)
    print(term.heading("What you can query") + "\n")
    print(describe_capabilities(caps))


def cmd_query(args) -> None:
    """Compute a value from the data (deterministic). E.g. fuel over a window."""
    globs, label = _resolve_window(args.source, args.vessel, args.days,
                                   getattr(args, "from"), args.to)
    if not globs:
        sys.exit(f"no data for '{args.source}' in the window.")
    present = [g for g in globs if glob.glob(g)]
    if not present:
        sys.exit(f"no parquet for '{args.source}' in window {label}.")

    if args.metric == "fuel_consumption":
        # Resolve the rate signal + unit from FIELD SEMANTICS (not hardcoded).
        # Prefer an explicit --signal; else find a signal marked aggregation=integral.
        kb = _kb(_asset_type(args))
        sig, unit = args.signal, args.unit or ""
        if not sig:
            for fs in kb.field_semantics(args.source):
                if fs.get("aggregation") == "integral":
                    sig, unit = fs["field"], fs.get("unit") or unit
                    break
        if not sig:
            sys.exit("No rate signal known for fuel consumption. Run `describe-fields` "
                     f"on '{args.source}' first, or pass --signal <fuel_rate_field>.")
        res = fuel_consumption(present, label, sig, unit=unit or "L")
    else:
        res = aggregate(present, args.signal, args.metric, label, unit=args.unit or "")
    print(term.heading(res.metric) + "\n")
    print(term.assistant(res.human()))


def cmd_plot(args) -> None:
    """Render a chart from the data: scatter (correlation) or timeseries."""
    globs, label = _resolve_window(args.source, args.vessel, args.days,
                                   getattr(args, "from"), args.to)
    present = [g for g in (globs or []) if glob.glob(g)]
    if not present:
        sys.exit(f"no data for '{args.source}' in the window.")

    if args.kind == "scatter":
        if not (args.x and args.y):
            sys.exit("scatter needs --x and --y signals.")
        out = config.ARTIFACTS_DIR / f"plot_{args.source}_{args.x}_vs_{args.y}.png"
        info = scatter(present, args.x, args.y, out)
        print(term.heading(f"Scatter: {args.y} vs {args.x}") + "\n")
        print(term.assistant(
            f"Pearson r={info['pearson']} over {info['points_total']:,} paired "
            f"readings ({info['points_plotted']:,} plotted)."))
        print(term.info(f"  saved: {out.relative_to(config.ROOT)}"))
    elif args.kind == "trend":
        if not args.signal:
            sys.exit("trend needs --signal (and optional --agg, --bucket)")
        out = config.ARTIFACTS_DIR / f"plot_{args.source}_{args.signal}_{args.agg}_{args.bucket}.png"
        info = aggregated_series(present, args.signal, out, bucket=args.bucket, agg=args.agg)
        print(term.heading(f"Trend: {args.agg} {args.signal} per {info['bucket']}") + "\n")
        print(term.assistant(f"{info['bins']} bins of {info['bucket']} over window {label}."))
        print(term.info(f"  saved: {out.relative_to(config.ROOT)}"))
    else:  # timeseries (raw)
        if not args.signals:
            sys.exit("timeseries needs --signals a,b,c")
        sigs = [s.strip() for s in args.signals.split(",")]
        out = config.ARTIFACTS_DIR / f"plot_{args.source}_timeseries.png"
        info = timeseries(present, sigs, out)
        print(term.heading("Timeseries: " + ", ".join(sigs)) + "\n")
        print(term.info(f"  saved: {out.relative_to(config.ROOT)}"))


def cmd_chat(args) -> None:
    """Single conversational surface. Orchestrates all phases, checks
    prerequisites, and runs missing ones on confirmation. Plain-language:
    ask for fields, values, plots, correlations, reasoning, or give corrections.
    """
    from ttmd.interpretation.orchestrator import Orchestrator
    from ttmd.chat_deps import ChatDeps

    provider = get_provider(meter=getattr(args, "usage", False))
    kb = _kb(_asset_type(args))
    deps = ChatDeps(_asset_type(args), provider, kb, vessel=args.vessel)

    sources = deps.all_sources()
    if not sources:
        sys.exit(f"No sources found for vessel '{args.vessel}'. Expected data under "
                 f"ship-data/{args.vessel}/<source>/logs/...")
    # home source (only matters for relationship/reasoning, which don't route
    # away): user override, else a source already discovered, else the most
    # signal-rich source. Data-driven — positional sources (ais/gps) have few
    # signals, so this naturally favors the sensor-heavy source without naming it.
    if args.source:
        home = args.source
    else:
        discovered = [s for s in sources if deps.has_discovery(s)]
        pool = discovered or sources
        home = max(pool, key=lambda s: len(deps.signals(s)))
    if home not in sources:
        sys.exit(f"Source '{home}' not found for '{args.vessel}'. "
                 f"Available: {', '.join(sources)}")

    session = Orchestrator(home, _asset_type(args), provider, kb, deps,
                           mode=getattr(args, "mode", "analyst"))

    others = [s for s in sources if s != home]
    print(term.heading(f"Chatting about '{args.vessel}' ({_asset_type(args)}) "
                       f"— home source '{home}'") +
          term.system(f"  [provider={type(provider).__name__}]"))
    print(term.system("Ask for fields, values, plots, correlations, reasoning — "
                      "or tell me a cause to remember. Type 'exit' to quit."))
    if others:
        print(term.system("Questions route automatically across this vessel's "
                          f"sources: {', '.join(deps.all_sources())}.\n"))
    else:
        print()
    if not deps.has_discovery(home):
        print(term.warn("Note: no analysis has run yet. Just ask — I'll analyze "
                        "the relevant source automatically the first time it's "
                        "needed (may take a moment).\n"))
    while True:
        try:
            msg = input(term.user("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("exit", "quit", "q"):
            break
        if msg.lower() in ("usage", "cost", "tokens") and hasattr(provider, "summary_line"):
            print(term.system("  " + provider.summary_line() + "\n"))
            continue
        print(term.system("  ...thinking"), end="\r", flush=True)
        reply = session.send(msg)
        print(" " * 20, end="\r")  # clear the thinking line
        print(f"{term.assistant('assistant>')} {term.assistant(reply)}\n")
    # on exit, if metering was on, print the session's LLM cost summary
    if hasattr(provider, "summary_line"):
        print(term.system(provider.summary_line()))


def cmd_ask(args) -> None:
    """Scope-gated entry point: refuses off-domain questions before any reasoning.

    (Seed of the future NL chat front-end. Demonstrates the domain boundary.)
    """
    provider = get_provider()
    decision = check_scope(args.question, provider)
    tag = term.info("IN") if decision.in_scope else term.warn("OUT")
    print(term.system(f"[scope: ") + tag + term.system(f" ({decision.reason})]"))
    if not decision.in_scope:
        print(term.warn(REFUSAL))
        return
    print(term.assistant("In scope. (The NL front-end would route this to a report/query.)"))


def cmd_correct(args) -> None:
    """Record a user correction/explanation for a relationship into the KB.

    --general-fact captures the GENERAL asset truth behind the correction, which
    generalizes to improve future reasoning about other relationships.
    """
    kb = _kb(_asset_type(args))
    apply_correction(kb, args.signal_a, args.signal_b, args.explanation,
                     general_fact=args.general_fact)
    print(f"Recorded: {args.signal_a} ~ {args.signal_b}")
    print(f"  explanation: {args.explanation}")
    if args.general_fact:
        print(f"  general fact (generalizes): {args.general_fact}")
    else:
        print("  tip: add --general-fact \"...\" to teach a reusable truth about the asset.")
    print(f"knowledge base now has {len(kb)} confirmed item(s): "
          f"{(config.ARTIFACTS_DIR / f'knowledge_{_asset_type(args)}.json').relative_to(config.ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ttmd", description="Relational Fingerprinting — zero-knowledge discovery.")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover", help="regimes + per-regime relationship graphs")
    d.add_argument("source", help="source name, e.g. engine / vibration / nmea")
    d.add_argument("--vessel", default=config.DEFAULT_VESSEL, help="vessel id")
    d.add_argument("--days", type=int, default=None,
                   help="use only the last N days of data (time window)")
    d.add_argument("--from", default=None, help="start date YYYY-MM-DD")
    d.add_argument("--to", default=None, help="end date YYYY-MM-DD")
    d.add_argument("--no-mi", action="store_true", help="skip mutual information (faster)")
    d.set_defaults(fn=cmd_discover)

    fu = sub.add_parser("fused", help="fuse sources on the time grid, cross-source discovery")
    fu.add_argument("sources", nargs="*", help="sources to fuse (default: all)")
    fu.add_argument("--vessel", default=config.DEFAULT_VESSEL, help="vessel id")
    fu.add_argument("--no-mi", action="store_true", help="skip mutual information (faster)")
    fu.set_defaults(fn=cmd_fused)

    bl = sub.add_parser("baseline", help="set a KNOWN-GOOD fingerprint to detect drift against")
    bl.add_argument("source", help="source name, e.g. engine")
    bl.add_argument("--vessel", default=config.DEFAULT_VESSEL, help="vessel id")
    bl.add_argument("--days", type=int, default=None, help="last N days as the baseline")
    bl.add_argument("--from", default=None, help="baseline start date YYYY-MM-DD")
    bl.add_argument("--to", default=None, help="baseline end date YYYY-MM-DD")
    bl.add_argument("--no-mi", action="store_true", help="skip mutual information (faster)")
    bl.set_defaults(fn=cmd_baseline)

    dt = sub.add_parser("detect", help="compare a window to the baseline and report drift")
    dt.add_argument("source", help="source name, e.g. engine")
    dt.add_argument("--vessel", default=config.DEFAULT_VESSEL, help="vessel id")
    dt.add_argument("--days", type=int, default=None, help="check the last N days")
    dt.add_argument("--from", default=None, help="window start date YYYY-MM-DD")
    dt.add_argument("--to", default=None, help="window end date YYYY-MM-DD")
    dt.add_argument("--no-mi", action="store_true", help="skip mutual information (faster)")
    dt.add_argument("--ae", action="store_true",
                    help="also run the optional autoencoder joint backend (nonlinear)")
    dt.set_defaults(fn=cmd_detect)

    rp = sub.add_parser("report", help="human-readable report from a discovery artifact")
    rp.add_argument("source", help="source name (or 'fused')")
    rp.set_defaults(fn=cmd_report)

    ip = sub.add_parser("interpret", help="report + LLM root-cause hypotheses (asset-aware)")
    ip.add_argument("source", help="source name (or 'fused')")
    ip.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    ip.add_argument("--no-docs", action="store_true", help="skip user documentation")
    ip.set_defaults(fn=cmd_interpret)

    ig = sub.add_parser("ingest-docs", help="pre-analyze user docs -> facts into KB")
    ig.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    ig.add_argument("--max-chunks", type=int, default=30,
                    help="cap chunks processed per document (cost control)")
    ig.set_defaults(fn=cmd_ingest_docs)

    rv = sub.add_parser("review-docs", help="review/promote/reject extracted doc facts")
    rv.add_argument("action", choices=["list", "promote", "reject"], default="list", nargs="?")
    rv.add_argument("fact_id", nargs="?", default=None, help="stable fact id from `list`")
    rv.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    rv.set_defaults(fn=cmd_review_docs)

    df = sub.add_parser("describe-fields", help="LLM proposes field meanings (protocol-aware)")
    df.add_argument("source", help="source, e.g. engine")
    df.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    df.add_argument("--vessel", default=config.DEFAULT_VESSEL)
    df.set_defaults(fn=cmd_describe_fields)

    pl = sub.add_parser("plot", help="render a chart: scatter (correlation) or timeseries")
    pl.add_argument("source", help="source, e.g. engine")
    pl.add_argument("kind", choices=["scatter", "trend", "timeseries"])
    pl.add_argument("--x", default=None, help="scatter x signal")
    pl.add_argument("--y", default=None, help="scatter y signal")
    pl.add_argument("--signal", default=None, help="trend signal")
    pl.add_argument("--agg", default="avg", help="trend aggregation: avg/min/max/median/sum")
    pl.add_argument("--bucket", default="hour", help="trend bucket: minute/hour/day")
    pl.add_argument("--signals", default=None, help="timeseries signals, comma-separated")
    pl.add_argument("--days", type=int, default=None)
    pl.add_argument("--from", default=None)
    pl.add_argument("--to", default=None)
    pl.add_argument("--vessel", default=config.DEFAULT_VESSEL)
    pl.set_defaults(fn=cmd_plot)

    cap = sub.add_parser("capabilities", help="list what you can query (available signals)")
    cap.add_argument("source", nargs="?", default=None, help="source (default: all)")
    cap.add_argument("--vessel", default=config.DEFAULT_VESSEL)
    cap.set_defaults(fn=cmd_capabilities)

    q = sub.add_parser("query", help="compute a value (e.g. total fuel over a window)")
    q.add_argument("source", help="source, e.g. engine")
    q.add_argument("metric", help="fuel_consumption | avg | min | max | sum | median | integral")
    q.add_argument("--signal", default=None, help="signal name (for generic aggregations)")
    q.add_argument("--unit", default=None)
    q.add_argument("--days", type=int, default=None)
    q.add_argument("--from", default=None)
    q.add_argument("--to", default=None)
    q.add_argument("--vessel", default=config.DEFAULT_VESSEL)
    q.set_defaults(fn=cmd_query)

    ch = sub.add_parser("chat", help="single conversational surface (orchestrates all phases)")
    ch.add_argument("vessel", nargs="?", default=config.DEFAULT_VESSEL,
                    help="vessel id (default: the only/first vessel). Chat is "
                         "vessel-scoped; questions auto-route across its sources.")
    ch.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    ch.add_argument("--source", default=None,
                    help="optional home source to start on (else auto-picked)")
    ch.add_argument("--mode", default="analyst",
                    help="presentation mode: operator (business, plain) / technician "
                         "(component-focused) / analyst (full detail, default). "
                         "Change mid-chat with 'switch to <mode> mode'.")
    ch.add_argument("--usage", action="store_true",
                    help="track LLM token usage + estimated cost (type 'usage' "
                         "mid-chat, and a summary prints on exit)")
    ch.set_defaults(fn=cmd_chat)

    ak = sub.add_parser("ask", help="scope-gated question entry (refuses off-domain)")
    ak.add_argument("question")
    ak.set_defaults(fn=cmd_ask)

    co = sub.add_parser("correct", help="record an expert explanation for a relationship")
    co.add_argument("signal_a")
    co.add_argument("signal_b")
    co.add_argument("explanation")
    co.add_argument("--general-fact", default=None,
                    help="reusable asset truth behind the correction (generalizes)")
    co.add_argument("--asset-type", default=None, help="asset type (else from sources.yaml)")
    co.set_defaults(fn=cmd_correct)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
