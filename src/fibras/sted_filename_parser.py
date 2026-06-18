"""Filename parsing for STED source manifests.

This is the only module that should infer grouping fields from raw STED
filenames. Downstream code reads the manifest fields instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?) \(Series (?P<series_index>\d+)\) \[(?P<bracket_index>\d+)\]\.tif$",
    re.IGNORECASE,
)
PN_RE = re.compile(r"^(PN\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedStedFilename:
    relative_path: str
    prefix: str
    inferred_pn: str
    inferred_round: str
    inferred_condition: str
    inferred_div: str
    series_index: str
    bracket_index: str
    acquisition_group: str
    biological_group_candidate: str
    parse_status: str


def parse_sted_filename(path: str | Path) -> ParsedStedFilename:
    """Parse known STED filenames without assigning biological meaning."""

    rel = Path(path).as_posix()
    name = Path(path).name
    match = FILENAME_RE.match(name)
    if not match:
        stem = Path(name).stem
        return ParsedStedFilename(
            relative_path=rel,
            prefix=stem,
            inferred_pn="unknown",
            inferred_round="unknown",
            inferred_condition="unknown",
            inferred_div="unknown",
            series_index="unknown",
            bracket_index="unknown",
            acquisition_group=stem,
            biological_group_candidate="unknown",
            parse_status="unparsed",
        )

    prefix = match.group("prefix")
    parts = prefix.split("_")
    pn = parts[0] if len(parts) > 0 and PN_RE.match(parts[0]) else "unknown"
    round_id = parts[1] if len(parts) > 1 else "unknown"
    condition = parts[2] if len(parts) > 2 else "unknown"
    div = parts[3] if len(parts) > 3 else "unknown"
    return ParsedStedFilename(
        relative_path=rel,
        prefix=prefix,
        inferred_pn=pn,
        inferred_round=round_id,
        inferred_condition=condition,
        inferred_div=div,
        series_index=match.group("series_index"),
        bracket_index=match.group("bracket_index"),
        acquisition_group=prefix,
        biological_group_candidate=pn,
        parse_status="parsed",
    )

