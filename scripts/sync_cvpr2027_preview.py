#!/usr/bin/env python3
"""Sync the CVPR-style preview body from the authoritative main manuscript."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "paper" / "main.tex"
PREVIEW = ROOT / "paper" / "cvpr2027_preview" / "main.tex"

MARKERS = (
    (r"\section{Introduction}", r"\typeout{CVPR_SECTION_Introduction: \thepage}"),
    (r"\section{Related Work}\label{sec:related-work}", r"\typeout{CVPR_SECTION_RelatedWork: \thepage}"),
    (r"\section{Method}", r"\typeout{CVPR_SECTION_Method: \thepage}"),
    (r"\section{Experiments}\label{sec:experiments}", r"\typeout{CVPR_SECTION_Experiments: \thepage}"),
    (r"\section{Limitations and conclusion}", r"\typeout{CVPR_SECTION_Limitations: \thepage}"),
    (r"\begin{thebibliography}{99}\small", r"\typeout{CVPR_BODY_PAGES_BEFORE_REFERENCES: \thepage}"),
)


def main() -> None:
    canonical = CANONICAL.read_text()
    existing_preview = PREVIEW.read_text()
    preamble = existing_preview.split(r"\begin{document}", 1)[0]
    body = r"\begin{document}" + canonical.split(r"\begin{document}", 1)[1]
    for anchor, marker in MARKERS:
        count = body.count(anchor)
        if count != 1:
            raise SystemExit(f"expected one occurrence of {anchor!r}, found {count}")
        body = body.replace(anchor, marker + "\n" + anchor, 1)
    PREVIEW.write_text(preamble + body)
    print(f"synced {PREVIEW.relative_to(ROOT)} from {CANONICAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
