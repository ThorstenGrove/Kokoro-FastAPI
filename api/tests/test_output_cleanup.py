"""Tests für die Löschfrist der /why/tts-Ausgaben (OUTPUT_RETENTION_HOURS).

Laufen im Image (pytest ist nicht installiert; bei Bedarf in einem Wegwerf-
Container mit `uv pip install pytest pytest-asyncio`).
"""

import os
import time

from api.src.core.config import Settings
from api.src.services.output_cleanup import cleanup_output_dir


def _touch(path, age_hours):
    with open(path, "wb"):
        pass
    t = time.time() - age_hours * 3600
    os.utime(path, (t, t))


def test_cleanup_removes_only_old_audio(tmp_path):
    _touch(tmp_path / "alt.mp3", 30)
    _touch(tmp_path / "neu.mp3", 1)
    _touch(tmp_path / "notiz.txt", 30)
    (tmp_path / "unterordner").mkdir()
    _touch(tmp_path / "unterordner" / "alt.mp3", 30)
    assert cleanup_output_dir(str(tmp_path), 24) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["neu.mp3", "notiz.txt", "unterordner"]
    assert (tmp_path / "unterordner" / "alt.mp3").exists()


def test_cleanup_seven_days(tmp_path):
    _touch(tmp_path / "sechs_tage.mp3", 6 * 24)
    _touch(tmp_path / "acht_tage.mp3", 8 * 24)
    assert cleanup_output_dir(str(tmp_path), 168) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["sechs_tage.mp3"]


def test_cleanup_disabled_with_zero(tmp_path):
    _touch(tmp_path / "alt.mp3", 30)
    assert cleanup_output_dir(str(tmp_path), 0) == 0
    assert (tmp_path / "alt.mp3").exists()


def test_cleanup_missing_dir(tmp_path):
    assert cleanup_output_dir(str(tmp_path / "fehlt"), 24) == 0


def test_settings_defaults_keep_old_behaviour():
    s = Settings(output_retention_hours=0, why_output_dir="/output")
    assert s.output_retention_hours == 0
    assert s.why_output_dir == "/output"
    assert Settings(output_retention_hours="168").output_retention_hours == 168
