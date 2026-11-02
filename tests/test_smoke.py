"""Smoke test ringan untuk modul inti backend."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import _json_safe, hitung_cer, hitung_wer, resolve_yolo_model_path  # noqa: E402


def test_yolo_weight_is_available() -> None:
    model_path = resolve_yolo_model_path()
    assert model_path is not None
    assert model_path.is_file()


def test_cer_wer_identical_text() -> None:
    assert hitung_cer("informasi nilai gizi", "informasi nilai gizi") == 0.0
    assert hitung_wer("informasi nilai gizi", "informasi nilai gizi") == 0.0


def test_json_safe_handles_nested_values() -> None:
    assert _json_safe({"a": [1, 2], "b": {"c": 3}}) == {"a": [1, 2], "b": {"c": 3}}
