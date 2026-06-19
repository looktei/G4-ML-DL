#!/usr/bin/env python3
"""Self-check for revised_reproduce_workflow_v.4 portability and logging behavior."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(workflow_dir: Path) -> None:
    workflow_dir = workflow_dir.resolve()
    check(workflow_dir.exists(), f"workflow dir not found: {workflow_dir}")
    check(not (workflow_dir / "reproduce_workflow_scripts").exists(), "v4 should not require reproduce_workflow_scripts/")
    wrappers = sorted(workflow_dir.glob("[0-9][0-9]_*.py"))
    check(len(wrappers) == 28, f"expected 28 standalone scripts, found {len(wrappers)}")

    proc = run([sys.executable, "-m", "py_compile", *map(str, wrappers)])
    check(proc.returncode == 0, f"py_compile failed\n{proc.stderr}")

    with tempfile.TemporaryDirectory(prefix="v4-selfcheck-") as tmp:
        copied = Path(tmp) / workflow_dir.name
        shutil.copytree(workflow_dir, copied, ignore=shutil.ignore_patterns("__pycache__", "logs"))
        for wrapper in sorted(copied.glob("[0-9][0-9]_*.py")):
            proc = run([sys.executable, wrapper.name, "--dry-run"], cwd=copied)
            combined = proc.stdout + proc.stderr
            check(proc.returncode == 0, f"dry-run failed for {wrapper.name}\n{combined}")
            check("Original script not found" not in combined, f"original script dependency error in {wrapper.name}")
            check("[START]" in combined and "[DONE]" in combined and "Logs:" in combined, f"missing status/log prints in {wrapper.name}")

        proc = run([sys.executable, "00_check_environment.py"], cwd=copied)
        combined = proc.stdout + proc.stderr
        check("Original script not found" not in combined, "00_check_environment still depends on original folder")
        check("[START]" in combined and "[DONE]" in combined, "00_check_environment missing status prints")
        status_logs = list((copied / "logs").glob("00_01_check_environment_*.status.json"))
        check(bool(status_logs), "00_check_environment did not write status json")
        for log in status_logs:
            check(re.search(r"_\d{8}_\d{6}_\d{6}\.status\.json$", log.name) is not None, f"log timestamp is not microsecond-resolution: {log.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_dir", nargs="?", default=".", help="Workflow folder to validate; default: current directory")
    args = parser.parse_args()
    validate(Path(args.workflow_dir))
    print("v4 self-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
