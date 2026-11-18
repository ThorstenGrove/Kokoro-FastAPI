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


# --- DE_G2P: mideki/misaki gegen das Modell-Vokabular ---------------------------

import json as _json

_VOCAB = _json.load(
    open(
        os.path.join(os.path.dirname(__file__), "..", "src", "models", "v1_0", "config.json"),
        encoding="utf-8",
    )
)["vocab"]

_SAETZE = [
    "Fünf Bücher über Glück und Mütter liegen im Küchenschrank.",
    "Heute ist es heiß, die Eule und die Mäuse schlafen. Der Pfau zupft an seinem Kopf.",
    "Am 3. Mai 2027 um 14:30 Uhr kostet der Kuchen 12,50 €.",
    "Beim Skateboarden schauen wir ein YouTube-Video am Computer.",
    "Ja, genau!",
]


@pytest.mark.parametrize("variant", ["misaki", "mideki", "mideki-word"])
def test_german_g2p_stays_in_vocab(variant):
    from api.src.services.text_processing.german_frontend import GermanG2P, unknown_symbols

    g2p = GermanG2P(variant, _VOCAB)
    for satz in _SAETZE:
        ps, _ = g2p(satz)
        assert ps
        assert unknown_symbols(ps, _VOCAB) == []
    assert g2p.dropped_total == 0


def test_all_mideki_overrides_map_into_vocab():
    from mideki.overrides import load_raw_overrides

    from api.src.services.text_processing.german_frontend import (
        MIDEKI_TO_TRAINING,
        apply_map,
        unknown_symbols,
    )

    raw = load_raw_overrides()
    bad = {}
    for section in ("brand", "en", "de_foreign", "de_native"):
        for key, value in raw.get(section, {}).items():
            unknown = unknown_symbols(apply_map(value, MIDEKI_TO_TRAINING), _VOCAB)
            if unknown:
                bad[key] = unknown
    assert bad == {}


def test_mideki_normalization_fixes_ordinal_and_keeps_numbers():
    from mideki.normalize import normalize_text_de as mideki_norm

    text = mideki_norm("Am 3. Mai um 14:30 Uhr kostet es 12,50 €.")
    assert "dritten Mai" in text
    assert "vierzehn Uhr dreißig" in text
    assert "zwölf Euro und fünfzig Cent" in text


def test_short_u_is_mapped_like_in_training():
    from api.src.services.text_processing.german_frontend import GermanG2P

    ps, _ = GermanG2P("misaki", _VOCAB)("Glück")
    assert "ʏ" not in ps and "y" in ps


def test_voice_gain_map():
    s = Settings(voice_gain_db="dm_martin=-6, df_victoria=0")
    gains = s.voice_gain_map()
    assert abs(gains["dm_martin"] - 0.501) < 0.01
    assert gains["df_victoria"] == 1.0
