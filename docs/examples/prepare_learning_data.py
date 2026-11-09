#!/usr/bin/env python3
"""Prepare the tutorial's 8/4/31 task split and record its source hashes.

This deterministic split is for learning the pipeline. It is not a scientifically
decontaminated train/validation/test split. Reference implementations are read as
bytes only; this script never imports torch or runs a reference model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import runpy
from pathlib import Path


ADD_TASK = "npukernelbench_level1_3_Add"
EXPECTED_TASKS = 43


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> tuple[bytes, list[tuple[int, dict]]]:
    """Read nonempty JSONL records while retaining their original line numbers."""
    raw = path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append((line_number, value))
    if not records:
        raise ValueError(f"{path}: no JSONL records")
    return raw, records


def prepare(project_root: Path, out: Path, case_set: str) -> dict:
    project_root = project_root.resolve()
    out = out.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output path already exists; refusing to overwrite: {out}")

    dataset = project_root / "ascendc-rl-datasets" / "NPUKernelBench"
    task_file = dataset / "operator_tasks.npukernelbench.jsonl"
    assets = dataset / case_set
    generator = (
        project_root / "ProRL-Agent-Server" / "deploy" / "ascend_operator"
        / "gen_ascendc_tasks.py"
    )
    source_bytes, records = read_jsonl(task_file)
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} source tasks, found {len(records)}")

    # run_name is deliberately not __main__: load _instruction without its CLI.
    generator_bytes = generator.read_bytes()
    instruction = runpy.run_path(str(generator))["_instruction"]
    if not callable(instruction):
        raise ValueError(f"{generator}: _instruction is not callable")

    tasks: dict[str, dict] = {}
    asset_manifest: dict[str, dict] = {}
    for source_line, record in records:
        metadata = record.get("metadata")
        op_name = metadata.get("op_name") if isinstance(metadata, dict) else None
        if not isinstance(op_name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", op_name):
            raise ValueError(f"{task_file}:{source_line}: invalid metadata.op_name")
        if op_name in tasks:
            raise ValueError(f"{task_file}:{source_line}: duplicate op_name {op_name}")

        reference_file = assets / f"{op_name}.py"
        case_file = assets / f"{op_name}.json"
        if not reference_file.is_file() or not case_file.is_file():
            raise ValueError(f"Missing .py/.json asset pair for {op_name} in {assets}")
        reference_bytes = reference_file.read_bytes()
        case_bytes, cases = read_jsonl(case_file)
        for case_line, case in cases:
            if not isinstance(case.get("inputs"), list):
                raise ValueError(f"{case_file}:{case_line}: expected an inputs list")

        updated = copy.deepcopy(record)
        updated["prompt"] = [{
            "role": "user",
            "content": instruction(op_name, f"input/{op_name}.py"),
        }]
        tasks[op_name] = updated
        asset_manifest[op_name] = {
            "source_task_line": source_line,
            "source_task_sha256": sha256(json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")),
            "reference": {"path": str(reference_file), "sha256": sha256(reference_bytes)},
            "cases": {
                "path": str(case_file), "sha256": sha256(case_bytes),
                "case_rows": len(cases),
            },
        }

    if ADD_TASK not in tasks:
        raise ValueError(f"Source task list does not contain {ADD_TASK}")
    ordered_names = [ADD_TASK] + [name for name in tasks if name != ADD_TASK]
    split_names = {
        "train": ordered_names[:8],
        "validation": ordered_names[8:12],
        "test": ordered_names[12:],
        "all43": ordered_names,
    }
    serialized = {}
    outputs = {}
    for split, names in split_names.items():
        payload = "".join(json.dumps(tasks[name], ensure_ascii=False) + "\n" for name in names)
        raw = payload.encode("utf-8")
        filename = f"{split}.jsonl"
        serialized[filename] = raw
        outputs[split] = {
            "file": filename, "sha256": sha256(raw), "task_count": len(names),
            "case_rows": sum(asset_manifest[name]["cases"]["case_rows"] for name in names),
            "op_names": names,
        }

    manifest = {
        "schema_version": 1,
        "purpose": "Teaching example for the AscendC agentic RL pipeline",
        "not_a_decontaminated_scientific_split": True,
        "limitations": [
            "8/4/31 is a teaching split, not a decontaminated scientific benchmark.",
            "No semantic deduplication or SFT contamination audit was performed.",
            "all43 overlaps all three splits and is not an independent held-out set.",
            "Asset checks are static JSON/file checks, not hardware validation.",
        ],
        "split_rule": "Move Add first; retain source order for all others; take 8/4/31.",
        "project_root": str(project_root),
        "case_set": case_set,
        "assets_directory": str(assets),
        "source_tasks": {
            "path": str(task_file), "sha256": sha256(source_bytes),
            "task_count": len(records),
        },
        "prompt_generator": {
            "path": str(generator), "sha256": sha256(generator_bytes),
            "function": "_instruction(op_name, model_rel)",
        },
        "metadata_policy": "Preserve each original record; replace prompt only.",
        "outputs": outputs,
        "assets": {name: asset_manifest[name] for name in ordered_names},
    }

    # Validate everything before creating an exclusive output directory.
    out.mkdir(parents=True, exist_ok=False)
    for filename, raw in serialized.items():
        with (out / filename).open("xb") as stream:
            stream.write(raw)
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2],
        help="Workspace containing the repositories (default: script's workspace)",
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--case-set", choices=("src", "src_simple"), default="src")
    args = parser.parse_args()
    try:
        manifest = prepare(args.project_root, args.out, args.case_set)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    counts = {name: value["task_count"] for name, value in manifest["outputs"].items()}
    print(json.dumps({"out": str(args.out.absolute()), "case_set": args.case_set, "counts": counts}))
    print("Teaching split only; inspect manifest.json before using it for an experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
