from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.release_prep_regression import _result_pass, _run_gate


GATE_SCRIPT = """\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-config", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--mode", choices=["pass", "payload_fail", "crash"], default="pass")
    args = parser.parse_args()

    if args.mode == "crash":
        print("forced crash", file=sys.stderr)
        return 3

    payload = {
        "overall_pass": args.mode == "pass",
        "failures": [] if args.mode == "pass" else ["synthetic failure"],
        "details": {"gate_config_seen": args.gate_config},
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _make_gate_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "scripts" / "fake_gate.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(GATE_SCRIPT, encoding="utf-8")
    return script_path


def test_result_pass_requires_zero_returncode() -> None:
    assert _result_pass({"overall_pass": True}, 1) is False
    assert _result_pass({"overall_pass": False}, 0) is False
    assert _result_pass({"overall_pass": True}, 0) is True


def test_run_gate_removes_stale_artifact_and_fails_on_crash(tmp_path: Path) -> None:
    script_path = _make_gate_script(tmp_path)
    artifact_path = tmp_path / "artifacts" / "gate.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"overall_pass": True}), encoding="utf-8")
    gate_config = tmp_path / "gate_thresholds.json"
    gate_config.write_text("{}", encoding="utf-8")

    result = _run_gate(
        gate_id="synthetic_gate",
        python_bin=sys.executable,
        script_path=script_path,
        script_args=["--mode", "crash"],
        cwd=tmp_path,
        artifact_path=artifact_path,
        gate_config=gate_config,
        help_cache={},
    )

    assert result["stale_artifact_removed"] is True
    assert result["gate_config_forwarded"] is True
    assert result["output_json_forwarded"] is True
    assert result["returncode"] == 3
    assert result["pass"] is False
    assert result["payload"] is None
    assert not artifact_path.exists()


def test_run_gate_forwards_options_and_reads_current_payload(tmp_path: Path) -> None:
    script_path = _make_gate_script(tmp_path)
    artifact_path = tmp_path / "artifacts" / "gate.json"
    gate_config = tmp_path / "gate_thresholds.json"
    gate_config.write_text("{}", encoding="utf-8")

    result = _run_gate(
        gate_id="synthetic_gate",
        python_bin=sys.executable,
        script_path=script_path,
        script_args=["--mode", "pass"],
        cwd=tmp_path,
        artifact_path=artifact_path,
        gate_config=gate_config,
        help_cache={},
    )

    assert result["pass"] is True
    assert result["returncode"] == 0
    assert result["gate_config_forwarded"] is True
    assert result["output_json_forwarded"] is True
    assert isinstance(result["payload"], dict)
    assert result["payload"]["overall_pass"] is True
    assert "--gate-config" in result["command"]
    assert "--output-json" in result["command"]
    assert artifact_path.exists()
