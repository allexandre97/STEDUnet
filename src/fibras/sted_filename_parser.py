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
DIV_RE = re.compile(r"^DIV(?P<div>\d+)$", re.IGNORECASE)
ALLOWED_TAU_ISOFORMS = {"3R", "4R"}
KNOWN_DISEASES = {"AD", "PID", "PSP", "CBD"}


@dataclass(frozen=True)
class ParsedStedFilename:
    relative_path: str
    prefix: str
    culture_id: str
    disease: str
    tau_isoform: str
    div: str
    div_token: str
    series_index: str
    bracket_index: str
    experimental_condition: str
    experimental_group_id: str
    acquisition_group: str
    parse_status: str

    @property
    def inferred_pn(self) -> str:
        return self.culture_id

    @property
    def inferred_round(self) -> str:
        return self.tau_isoform

    @property
    def inferred_condition(self) -> str:
        return self.disease

    @property
    def inferred_div(self) -> str:
        return self.div_token

    @property
    def biological_group_candidate(self) -> str:
        return self.experimental_group_id


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
            culture_id="unknown",
            disease="unknown",
            tau_isoform="unknown",
            div="unknown",
            div_token="unknown",
            series_index="unknown",
            bracket_index="unknown",
            experimental_condition="unknown",
            experimental_group_id="unknown",
            acquisition_group=stem,
            parse_status="unparsed",
        )

    prefix = match.group("prefix")
    parts = prefix.split("_")
    culture_id = parts[0].upper() if len(parts) > 0 and PN_RE.match(parts[0]) else "unknown"
    tau_token = parts[1].upper() if len(parts) > 1 else "unknown"
    tau_isoform = tau_token if tau_token in ALLOWED_TAU_ISOFORMS else "unknown"
    disease_token = parts[2].upper() if len(parts) > 2 else "unknown"
    disease = disease_token if disease_token in KNOWN_DISEASES else disease_token
    div_token = parts[3].upper() if len(parts) > 3 else "unknown"
    div = parse_div(div_token)
    experimental_condition = f"{disease}_{tau_isoform}" if "unknown" not in {disease, tau_isoform} else "unknown"
    experimental_group_id = (
        f"{culture_id}_{experimental_condition}_DIV{int(div):02d}"
        if "unknown" not in {culture_id, experimental_condition, div}
        else "unknown"
    )
    parse_status = "parsed"
    if tau_token != "unknown" and tau_isoform == "unknown":
        parse_status = "invalid_tau_isoform"
    if div_token != "unknown" and div == "unknown":
        parse_status = "invalid_div"
    return ParsedStedFilename(
        relative_path=rel,
        prefix=prefix,
        culture_id=culture_id,
        disease=disease,
        tau_isoform=tau_isoform,
        div=div,
        div_token=div_token,
        series_index=match.group("series_index"),
        bracket_index=match.group("bracket_index"),
        experimental_condition=experimental_condition,
        experimental_group_id=experimental_group_id,
        acquisition_group=prefix,
        parse_status=parse_status,
    )


def parse_div(token: str) -> str:
    match = DIV_RE.match(token)
    if not match:
        return "unknown"
    return str(int(match.group("div")))
