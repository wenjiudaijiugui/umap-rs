from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional


THRESHOLDS_PATH = Path(__file__).with_name("gate_thresholds.json")


def load_gate_config(gate: str, config_path: Optional[str] = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else THRESHOLDS_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload.get("gates", payload)
    if not isinstance(root, dict):
        raise RuntimeError(f"invalid gate config layout in {path}")
    config = root.get(gate)
    if not isinstance(config, dict):
        raise RuntimeError(f"missing gate config for {gate!r} in {path}")
    return config


def gate_report(
    *,
    gate: str,
    strict: bool,
    overall_pass: bool,
    thresholds: Mapping[str, Any],
    failures: list[str],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": gate,
        "strict": strict,
        "overall_pass": overall_pass,
        "thresholds": dict(thresholds),
        "failures": list(failures),
        "timestamp_unix": int(time.time()),
        "details": dict(details),
    }


def emit_report(report: Mapping[str, Any], output_json: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output_json:
        Path(output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
