"""Human-readable data-intelligence reports from discovery artifacts.

Turns the machine output (regimes, relationship edges, dCor/Pearson values) into
plain-language descriptions. Deterministic/template-based: it describes ONLY what
was measured (structure), never inventing meaning the data does not contain.
"""
from .describe import describe_discovery, render_markdown

__all__ = ["describe_discovery", "render_markdown"]
