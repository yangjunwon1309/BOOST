"""Unit checks for the D405 topology CLI; no hardware/pyrealsense needed."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_d405_stereo_basis.py"
SPEC = importlib.util.spec_from_file_location("check_d405_stereo_basis", MODULE_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_vector_norm_mm() -> None:
    assert checker._vector_norm_mm((0.003, 0.004, 0.0)) == 5.0


def test_d405_product_id_is_normalized() -> None:
    assert checker.D405_PRODUCT_ID == "0B5B"


def test_remote_args_do_not_repeat_ssh_destination(monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, input, text):
        captured["command"] = command
        captured["source"] = input
        captured["text"] = text
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(checker.subprocess, "run", fake_run)
    args = argparse.Namespace(serial=["one"], json=True, backend="v4l2", require_two=True)
    assert checker._run_on_ssh("url@thor", args) == 0
    assert captured["command"] == [
        "ssh", "url@thor", "python3", "-", "--serial", "one", "--json", "--backend", "v4l2", "--require-two"
    ]
    assert "def main()" in captured["source"]
    assert captured["text"] is True
