"""Tests für den deutschen Container: Löschfrist, Einstellungen, deutscher G2P.

Laufen im Image (pytest ist nicht installiert; bei Bedarf in einer Kopie mit
`uv pip install pytest pytest-asyncio`). Der G2P-Test braucht misaki/espeak.
"""

import os
import time

import pytest

from api.src.core.config import Settings
from api.src.services.output_cleanup import cleanup_output_dir
from api.src.services.text_processing.german_g2p import normalize_text_de


def _touch(path, age_hours):
    with open(path, "wb"):
        pass
    t = time.time() - age_hours * 3600
    os.utime(path, (t, t))


def test_cleanup_removes_only_old_audio(tmp_path):
    _touch(tmp_path / "alt.mp3", 30)
    _touch(tmp_path / "neu.mp3", 1)
    _touch(tmp_path / "notiz.txt", 30)
    assert cleanup_output_dir(str(tmp_path), 24) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["neu.mp3", "notiz.txt"]


def test_cleanup_disabled_with_zero(tmp_path):
    _touch(tmp_path / "alt.mp3", 30)
    assert cleanup_output_dir(str(tmp_path), 0) == 0
    assert (tmp_path / "alt.mp3").exists()


def test_voice_model_map_and_lang_codes():
    s = Settings(
        voice_models="dm_martin=/m/martin.pth, df_victoria=/m/victoria.pth",
        allowed_lang_codes="d, D",
    )
    assert s.voice_model_map() == {
        "dm_martin": "/m/martin.pth",
        "df_victoria": "/m/victoria.pth",
    }
    assert s.allowed_lang_code_set() == {"d"}
    # Im DE-Image sind VOICE_MODELS/ALLOWED_LANG_CODES gesetzt; ausdrücklich leeren
    empty = Settings(voice_models=None, allowed_lang_codes=None)
    assert empty.voice_model_map() == {}
    assert empty.allowed_lang_code_set() == set()


def test_voice_model_map_rejects_garbage():
    with pytest.raises(ValueError):
        Settings(voice_models="dm_martin").voice_model_map()


def test_german_normalization():
    assert "fünftausend" in normalize_text_de("Er wiegt 5000 Kilogramm.")
    assert "zum Beispiel" in normalize_text_de("z.B. ein Tablet")


def test_german_g2p_produces_phonemes():
    from api.src.services.text_processing.german_g2p import DEG2P

    ps, _ = DEG2P()("Warum ist der Himmel blau?")
    assert ps and len(ps) > 10
