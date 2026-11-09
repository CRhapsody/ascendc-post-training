#!/usr/bin/env python3
"""Inspect SFT records and optionally export explicitly selected examples.

Accept JSON arrays, JSONL, a single JSON object, or a .zst containing one of
these formats. Uses only the Python standard library and the system zstd
decoder. Never execute code or tool calls found in the dataset.

Alpaca records become system/history/current-user/assistant messages. Existing
messages and their record-level fields (including tools and metadata) are
preserved, including role/content/tool_calls/tool_call_id within messages. Tool,
user, and system messages are context, not supervised output targets. This
tool does NOT tokenize records or generate loss masks: the training recipe
must implement assistant-target masking and the model's tool-call template.
It flags repeated records/tool-ID sequences but never deduplicates them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
ROLES = {"system", "developer", "user", "assistant", "tool"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(record) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def text_chars(value) -> int:
    """Count Unicode characters in string leaves, never tokens or words."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(text_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(text_chars(item) for item in value.values())
    return 0


def load_records(path: Path):
    raw = path.read_bytes()
    if raw.lstrip().startswith(LFS_HEADER):
        raise ValueError("Input is a Git LFS pointer; download its real object before inspection.")
    decoded = raw
    compression = None
    if path.suffix.lower() == ".zst":
        decoder = shutil.which("zstd")
        if not decoder:
            raise ValueError("Cannot decode .zst: system zstd was not found. Install zstd or supply a decoded file.")
        result = subprocess.run(
            [decoder, "--decompress", "--stdout", "--", str(path)],
            capture_output=True, check=False,
        )
        if result.returncode:
            raise ValueError("zstd decompression failed: " + result.stderr.decode("utf-8", errors="replace")[-500:])
        decoded = result.stdout
        compression = "zstd"
    if decoded.lstrip().startswith(LFS_HEADER):
        raise ValueError("Decoded content is a Git LFS pointer, not dataset records.")
    text = decoded.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("Input contains no records.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        source_format = "jsonl"
    else:
        if isinstance(value, list):
            records, source_format = value, "json_array"
        elif isinstance(value, dict):
            records, source_format = [value], "json_object"
        else:
            raise ValueError("Top-level JSON must be an array of records or a record object.")
    if not records:
        raise ValueError("Input contains no records.")
    return records, {
        "path": str(path), "sha256": digest(raw), "bytes": len(raw),
        "decoded_sha256": digest(decoded), "decoded_bytes": len(decoded),
        "decoded_characters": len(text), "format": source_format, "compression": compression,
    }


def normalize_record(record, emit):
    """Validate a record; return its schema, message list, and ordered call IDs."""
    if not isinstance(record, dict):
        emit("record_not_object", "Record must be a JSON object.")
        return "unknown", [], []
    if "messages" not in record:
        bad = False
        for key in ("instruction", "input", "output", "system"):
            if not isinstance(record.get(key), str):
                emit("alpaca_field_type", f"{key} must be a string.")
                bad = True
        history = record.get("history")
        if not isinstance(history, list):
            emit("alpaca_history_type", "history must be a list of two-string pairs.")
            bad = True
        else:
            for position, pair in enumerate(history):
                if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
                    emit("alpaca_history_pair", f"history[{position}] must be [user_string, assistant_string].")
                    bad = True
        if bad:
            return "alpaca", [], []
        messages = []
        if record["system"]:
            messages.append({"role": "system", "content": record["system"]})
        for user, assistant in history:
            messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        messages.extend([
            {"role": "user", "content": "\n\n".join(record[k] for k in ("instruction", "input") if record[k])},
            {"role": "assistant", "content": record["output"]},
        ])
        return "alpaca", messages, []

    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        emit("messages_type", "messages must be a nonempty list.")
        return "messages", [], []
    seen, pending, replied = set(), set(), set()
    sequence = []
    for position, message in enumerate(messages):
        label = f"messages[{position}]"
        if not isinstance(message, dict):
            emit("message_not_object", f"{label} must be an object.")
            continue
        role = message.get("role")
        if not isinstance(role, str) or role not in ROLES:
            emit("message_role", f"{label} has an unsupported or missing role.")
        if message.get("content") is not None and not isinstance(message["content"], (str, list)):
            emit("message_content", f"{label}.content must be a string, content-block list, or null.")
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            emit("tool_calls_type", f"{label}.tool_calls must be a list or null.")
            calls = []
        if calls and role != "assistant":
            emit("tool_calls_role", f"{label}: tool calls must belong to an assistant message.")
        for call_position, call in enumerate(calls):
            location = f"{label}.tool_calls[{call_position}]"
            if not isinstance(call, dict):
                emit("tool_call_object", f"{location} must be an object.")
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                emit("tool_call_id", f"{location}.id must be a nonempty string.")
            else:
                sequence.append(call_id)
                if call_id in seen:
                    emit("duplicate_tool_call_id", f"{location}: call ID is reused within this record.")
                seen.add(call_id)
                pending.add(call_id)
            function = call.get("function")
            if call.get("type", "function") != "function" or not isinstance(function, dict):
                emit("tool_function", f"{location} must describe a function tool call.")
                continue
            if not isinstance(function.get("name"), str) or not function["name"]:
                emit("tool_function_name", f"{location}.function.name must be a nonempty string.")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                emit("tool_arguments_type", f"{location}.function.arguments must be a JSON string.")
            else:
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    emit("tool_arguments_json", f"{location}.function.arguments is not valid JSON.")
        reply_id = message.get("tool_call_id")
        if role == "tool":
            if not isinstance(reply_id, str) or not reply_id:
                emit("tool_reply_id", f"{label}: tool reply requires a nonempty tool_call_id.")
            elif reply_id in replied:
                emit("duplicate_tool_reply", f"{label}: a call already received a tool reply.")
            elif reply_id not in pending:
                emit("orphan_tool_reply", f"{label}: reply has no preceding unmatched call.")
            else:
                pending.remove(reply_id)
                replied.add(reply_id)
        elif reply_id not in (None, ""):
            emit("non_tool_reply_id", f"{label}: non-tool message has a nonempty tool_call_id.")
    if pending:
        emit("missing_tool_reply", f"{len(pending)} tool call(s) have no reply by the end of this record.")
    return "messages", messages, sequence


def inspect(records, source):
    schemas, roles, content_chars, issue_counts = Counter(), Counter(), Counter(), Counter()
    warnings, invalid, normalized, record_hashes = [], set(), [], []
    exact_groups, tool_groups = defaultdict(list), defaultdict(list)
    sequence_lengths = {}
    argument_chars = tool_calls = 0
    for index, record in enumerate(records):
        def emit(code, detail):
            invalid.add(index)
            issue_counts[code] += 1
            if len(warnings) < 100:
                warnings.append({"index": index, "code": code, "detail": detail})

        schema, messages, sequence = normalize_record(record, emit)
        schemas[schema] += 1
        normalized.append(dict(record) if schema == "messages" else {"messages": messages})
        record_hash = digest(canonical(record))
        record_hashes.append(record_hash)
        exact_groups[record_hash].append(index)
        if sequence:
            signature = digest(canonical(sequence))
            tool_groups[signature].append(index)
            sequence_lengths[signature] = len(sequence)
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            role_key = role if isinstance(role, str) else "<invalid>"
            roles[role_key] += 1
            content_chars[role_key] += text_chars(message.get("content"))
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                tool_calls += len(calls)
                for call in calls:
                    if isinstance(call, dict) and isinstance(call.get("function"), dict):
                        argument_chars += text_chars(call["function"].get("arguments"))

    duplicates = [{"canonical_sha256": key, "indices": indices}
                  for key, indices in exact_groups.items() if len(indices) > 1]
    repeated_sequences = [{"sequence_sha256": key, "call_count": sequence_lengths[key], "indices": indices}
                          for key, indices in tool_groups.items() if len(indices) > 1]
    dataset_warnings = []
    if duplicates:
        dataset_warnings.append({
            "code": "identical_records", "group_count": len(duplicates),
            "detail": "Parsed records repeat exactly. They were retained; audit their provenance before splitting.",
        })
    if repeated_sequences:
        dataset_warnings.append({
            "code": "shared_tool_call_id_sequences", "group_count": len(repeated_sequences),
            "detail": "Different records reuse an ordered tool-ID sequence. Review their common source before assigning splits; this alone does not prove leakage.",
        })
    report = {
        "input": source,
        "record_count": len(records), "schema_counts": dict(schemas),
        "valid_record_count": len(records) - len(invalid), "invalid_record_count": len(invalid),
        "message_roles": dict(roles), "message_content_characters_by_role": dict(content_chars),
        "tool_call_count": tool_calls, "tool_argument_characters": argument_chars,
        "character_count_definition": "Unicode characters in content string leaves; not tokens or linguistic word counts.",
        "validation_issue_counts": dict(issue_counts), "warnings": warnings,
        "dataset_warnings": dataset_warnings,
        "warnings_omitted": max(0, sum(issue_counts.values()) - len(warnings)),
        "canonical_unique_records": len(exact_groups), "identical_record_groups": duplicates,
        "records_with_tool_call_ids": sum(map(len, tool_groups.values())),
        "unique_tool_call_id_sequences": len(tool_groups), "repeated_tool_call_id_sequence_groups": repeated_sequences,
        "leakage_note": "Repeated tool-ID sequences are a shared-provenance hint, not proof of semantic equivalence or leakage. Keep related records together when auditing splits; no records were removed.",
        "training_target_policy": "Only assistant output is a potential target. Tool/user/system/developer messages are context. No tokenizer, chat template, token IDs, or loss masks are produced.",
    }
    return report, normalized, record_hashes, invalid


def export_selected(out, indices, records, normalized, record_hashes, invalid, report):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output path already exists; refusing to overwrite: {out}")
    for index in indices:
        if index < 0 or index >= len(records):
            raise ValueError(f"Index {index} is out of range for {len(records)} records.")
        if index in invalid:
            raise ValueError(f"Selected record {index} failed validation; inspect warnings before exporting.")
    exports = {
        "records.raw.jsonl": b"".join(json.dumps(records[i], ensure_ascii=False).encode("utf-8") + b"\n" for i in indices),
        "messages.jsonl": b"".join(json.dumps(normalized[i], ensure_ascii=False).encode("utf-8") + b"\n" for i in indices),
    }
    manifest = {
        "schema_version": 1, "input": report["input"], "selected_indices": indices,
        "records": [{"original_index": i, "raw_record_canonical_sha256": record_hashes[i],
                     "messages_canonical_sha256": digest(canonical(normalized[i]))} for i in indices],
        "hash_definition": "Record hashes use UTF-8 JSON with sorted keys, ensure_ascii=False and compact separators; input and export hashes use exact file bytes.",
        "conversion_rules": [
            "Alpaca: optional nonempty system; history user/assistant pairs in source order; current user; final assistant output.",
            "Current user joins nonempty instruction and input with exactly two newline characters; text is not stripped.",
            "Existing messages records preserve all top-level fields, including tools and metadata, plus every message field and tool call/reply ID.",
            "records.raw.jsonl preserves parsed record values, not original whitespace or array delimiters.",
            "Tool/user/system/developer messages are context, not training targets. Assistant masking and tokenization must be implemented by the trainer.",
            "No automatic deduplication, semantic cleanup, truncation, tool execution, or data-code execution.",
        ],
        "outputs": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in exports.items()},
        "full_file_statistics": report,
    }
    out.mkdir(parents=True, exist_ok=False)
    for filename, raw in exports.items():
        with (out / filename).open("xb") as stream:
            stream.write(raw)
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--index", type=int, action="append", default=[], help="Zero-based record index; repeat for multiple records")
    parser.add_argument("--out", type=Path, help="New output directory; requires explicit --index values")
    args = parser.parse_args()
    if bool(args.index) != (args.out is not None):
        parser.error("Use --out together with one or more explicit --index values; inspection alone exports nothing.")
    try:
        if args.out is not None and (args.out.exists() or args.out.is_symlink()):
            raise FileExistsError(f"Output path already exists; refusing to overwrite: {args.out}")
        records, source = load_records(args.input.resolve())
        report, normalized, record_hashes, invalid = inspect(records, source)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.out is not None:
            export_selected(args.out.absolute(), args.index, records, normalized, record_hashes, invalid, report)
            print(f"Exported {len(args.index)} explicitly selected record(s) to {args.out.absolute()}", file=sys.stderr)
        return 1 if invalid else 0
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
