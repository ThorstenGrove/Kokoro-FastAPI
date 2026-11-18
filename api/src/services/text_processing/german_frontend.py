"""Deutscher G2P für die kikiri-Stimmen Martin/Victoria, umschaltbar per DE_G2P.

Trainiert wurden die Checkpoints mit misaki.espeak.EspeakG2P(language="de") im
Satzmodus (ganzer Text an espeak-ng, espeakng-loader 0.2.4) und danach "ʏ"→"y",
weil "ʏ" nicht im Kokoro-Vokabular steht (semidark/kikiri-tts,
scripts/prepare_dataset.py und docs/ARCHITECTURE.md). Alles, was hier an das
Modell geht, muss in diesem Symbolsatz bleiben.

Varianten (DE_G2P):
- "misaki": DEG2P aus misaki PR #97 (german_g2p.py): Normalisierung + kleines
  Override-Lexikon + EspeakG2P. Dazu jetzt "ʏ"→"y" (vorher still verworfen).
- "mideki": Normalisierung und Override-Lexikon aus mideki (ichDaheim/mideki,
  Apache-2.0), die Phonemisierung bleibt beim Trainings-G2P (EspeakG2P im
  Satzmodus). Die Override-Werte sind in mideki-Symbolen geschrieben und werden
  auf den Trainingssatz abgebildet (MIDEKI_TO_TRAINING).
- "mideki-word": mideki komplett (Wortmodus, eigene Symbole), nur auf das
  Vokabular abgebildet. Nur zum Vergleich: Wortmodus-Betonung (jedes Wort
  betont) und die Symbole kennt das Modell aus dem Training nicht.

Jede Variante zählt Symbole, die nicht im Modell-Vokabular stehen und von
Kokoro still verworfen würden, und loggt nur die Anzahl und die Symbole.
"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from loguru import logger

# Wie im Training (kikiri-tts prepare_dataset.py: PHONEME_FIXUPS)
TRAINING_FIXUPS: Tuple[Tuple[str, str], ...] = (("ʏ", "y"),)

# mideki-Symbole, die EspeakG2P aus misaki nicht erzeugt, auf dessen Ausgabe
# abgebildet. "ʋ" = mideki-Kompression von ɔ^ø ("eu"), "ꭧ" = "pf".
# "æ" (nur in handgeschriebenen englischen Overrides) gibt es im Trainings-G2P
# praktisch nicht; nächster trainierter Laut ist "ɛ".
MIDEKI_TO_TRAINING: Tuple[Tuple[str, str], ...] = (
    ("ʋ", "ɔø"),
    ("ꭧ", "pf"),
    ("æ", "ɛ"),
) + TRAINING_FIXUPS

VARIANTS = ("misaki", "mideki", "mideki-word")


def apply_map(ps: str, mapping: Iterable[Tuple[str, str]]) -> str:
    for old, new in mapping:
        ps = ps.replace(old, new)
    return ps


def unknown_symbols(ps: str, vocab: Dict[str, int]) -> List[str]:
    """Symbole (Zeichen) außerhalb des Vokabulars; Leerzeichen zählt als bekannt."""
    return [ch for ch in ps if ch not in vocab]


class _MidekiHybridG2P:
    """mideki-Normalisierung und -Overrides, Phonemisierung wie im Training."""

    def __init__(self):
        from misaki.espeak import EspeakG2P
        from mideki.g2p import render
        from mideki.normalize import normalize_text_de
        from mideki.overrides import OVERRIDE_WORD_RE, find_phrase_override

        self._espeak = EspeakG2P(language="de")
        self._render = render
        self._normalize = normalize_text_de
        self._word_re = OVERRIDE_WORD_RE
        self._find = find_phrase_override

    def _span(self, text: str) -> str:
        return self._espeak(text)[0] if text.strip() else ""

    def __call__(self, text: str) -> Tuple[str, None]:
        normalized = self._normalize(text)
        parts: List[str] = []
        cursor = 0
        matches = list(self._word_re.finditer(normalized))
        i = 0
        while i < len(matches):
            end_index, phonemes = self._find(normalized, matches, i)
            if phonemes is None:
                i += 1
                continue
            parts.append(self._span(normalized[cursor:matches[i].start()]))
            parts.append(apply_map(phonemes, MIDEKI_TO_TRAINING))
            cursor = matches[end_index].end()
            i = end_index + 1
        if cursor == 0:
            return self._span(normalized), None
        parts.append(self._span(normalized[cursor:]))
        return self._render(parts), None


class _MidekiWordG2P:
    def __init__(self):
        from mideki import G2P

        self._g2p = G2P(mode="word")

    def __call__(self, text: str) -> Tuple[str, None]:
        ps, _ = self._g2p(text)
        return apply_map(ps, MIDEKI_TO_TRAINING), None


class GermanG2P:
    """Wählt die Variante, wendet die Trainings-Fixups an und zählt Verluste."""

    def __init__(self, variant: str, vocab: Optional[Dict[str, int]] = None):
        if variant not in VARIANTS:
            raise ValueError(f"DE_G2P must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.vocab = vocab or {}
        if variant == "misaki":
            from .german_g2p import DEG2P

            self._inner: Callable[[str], Tuple[str, None]] = DEG2P()
        elif variant == "mideki":
            self._inner = _MidekiHybridG2P()
        else:
            self._inner = _MidekiWordG2P()
        self.dropped_total = 0
        logger.info(f"German G2P: {variant}")

    def __call__(self, text: str) -> Tuple[str, None]:
        ps, _ = self._inner(text)
        ps = apply_map(ps or "", TRAINING_FIXUPS)
        if self.vocab:
            unknown = unknown_symbols(ps, self.vocab)
            if unknown:
                # Nur Anzahl und Symbole, nie Text
                self.dropped_total += len(unknown)
                logger.warning(
                    f"German G2P ({self.variant}): {len(unknown)} symbol(s) outside "
                    f"model vocab would be dropped: {sorted(set(unknown))}"
                )
        return ps, None


def strip_unknown(ps: str, vocab: Dict[str, int]) -> str:
    """Nur für Tests/Metriken: was Kokoro tatsächlich verwendet."""
    return "".join(ch for ch in ps if ch in vocab)

