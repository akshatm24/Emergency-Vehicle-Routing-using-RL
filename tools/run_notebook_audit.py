#!/usr/bin/env python3
"""Execute a notebook and write per-cell timing/error metadata."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


FAILURE_PATTERNS = [
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"\binf(?:inity)?\b", re.IGNORECASE),
    re.compile(r"empty\s+(?:dataframe|result|list|array)", re.IGNORECASE),
    re.compile(r"shape.*mismatch", re.IGNORECASE),
    re.compile(r"0\s+hospitals", re.IGNORECASE),
]


class TimingNotebookClient(NotebookClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = {}

    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        start = time.perf_counter()
        try:
            return await super().async_execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )
        finally:
            self.timings[cell_index] = time.perf_counter() - start


def source_preview(cell):
    src = "".join(cell.get("source", ""))
    return " ".join(src.strip().split())[:240]


def output_text(output):
    chunks = []
    if output.get("output_type") == "stream":
        chunks.append(output.get("text", ""))
    if "text/plain" in output.get("data", {}):
        chunks.append(output["data"]["text/plain"])
    if "text" in output:
        chunks.append(output.get("text", ""))
    if output.get("output_type") == "error":
        chunks.append(output.get("ename", ""))
        chunks.append(output.get("evalue", ""))
        chunks.extend(output.get("traceback", []))
    return "\n".join(chunks)


def build_report(nb, timings, elapsed_s):
    report = {
        "elapsed_s": elapsed_s,
        "cell_count": len(nb.cells),
        "code_cell_count": sum(1 for c in nb.cells if c.cell_type == "code"),
        "errors": [],
        "warnings": [],
        "cells": [],
    }

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        text = "\n".join(output_text(o) for o in cell.get("outputs", []))
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        warning_hits = sorted({p.pattern for p in FAILURE_PATTERNS if p.search(text)})
        cell_record = {
            "index": idx,
            "execution_count": cell.get("execution_count"),
            "duration_s": timings.get(idx),
            "source_preview": source_preview(cell),
            "error_count": len(errors),
            "warning_patterns": warning_hits,
        }
        report["cells"].append(cell_record)

        for err in errors:
            report["errors"].append(
                {
                    "cell_index": idx,
                    "duration_s": timings.get(idx),
                    "source_preview": source_preview(cell),
                    "ename": err.get("ename"),
                    "evalue": err.get("evalue"),
                    "traceback_tail": err.get("traceback", [])[-6:],
                }
            )
        if warning_hits:
            report["warnings"].append(
                {
                    "cell_index": idx,
                    "duration_s": timings.get(idx),
                    "source_preview": source_preview(cell),
                    "patterns": warning_hits,
                    "output_tail": text[-2000:],
                }
            )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=-1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(args.notebook, as_version=4)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    client = TimingNotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(args.outdir.resolve())}},
    )

    start = time.perf_counter()
    client.execute()
    elapsed_s = time.perf_counter() - start

    stem = args.notebook.stem.replace(" ", "_").replace("(", "").replace(")", "")
    executed_path = args.outdir / f"{stem}.executed.ipynb"
    report_path = args.outdir / f"{stem}.audit.json"
    nbformat.write(nb, executed_path)
    report = build_report(nb, client.timings, elapsed_s)
    report["notebook"] = str(args.notebook)
    report["executed_notebook"] = str(executed_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "notebook": str(args.notebook),
        "elapsed_s": round(elapsed_s, 2),
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
        "executed_notebook": str(executed_path),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
