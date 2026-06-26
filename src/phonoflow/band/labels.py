"""High-symmetry label normalization for phonon band paths."""

from __future__ import annotations

GAMMA = "\N{GREEK CAPITAL LETTER GAMMA}"
_GAMMA_ALIASES = {"G", "GAMMA", "\\GAMMA", GAMMA}


def normalize_band_label(label: str | None) -> str:
    """Normalize one high-symmetry label fragment without merging boundaries."""

    clean = "" if label is None else str(label).strip()
    if not clean:
        return ""
    if clean.upper() in _GAMMA_ALIASES or clean == GAMMA:
        return GAMMA
    return clean


def merge_boundary_labels(labels: list[str]) -> str:
    """Merge same-position labels with ``|`` and collapse exact duplicates."""

    merged: list[str] = []
    for label in labels:
        normalized = normalize_band_label(label)
        if not normalized:
            continue
        if not merged or merged[-1] != normalized:
            merged.append(normalized)
    if len(merged) == 2 and merged[0] == merged[1]:
        return merged[0]
    return "|".join(merged)


def format_band_label(label: str | None) -> str:
    """Format a label or compound boundary label for plots and metadata."""

    parts = [part for part in str(label or "").split("|")]
    return merge_boundary_labels(parts)


def collapse_tick_labels(ticks: list[float], labels: list[str]) -> tuple[list[float], list[str]]:
    """Collapse same-position explicit labels into formatted plot ticks."""

    tick_positions: list[float] = []
    grouped_labels: list[list[str]] = []
    for tick, label in zip(ticks, labels, strict=False):
        formatted = format_band_label(label)
        if not formatted:
            continue
        if tick_positions and abs(tick_positions[-1] - float(tick)) < 1e-8:
            grouped_labels[-1].append(formatted)
        else:
            tick_positions.append(float(tick))
            grouped_labels.append([formatted])
    return tick_positions, [merge_boundary_labels(labels) for labels in grouped_labels]
