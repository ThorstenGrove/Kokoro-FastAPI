# Fremdcode: deutscher G2P

`api/src/services/text_processing/german_g2p.py` und `de_overrides.json` stammen aus
misaki (https://github.com/hexgrad/misaki), Pull Request #97 „German G2P with text
normalization“, Commit `bbaf917e7bf7fbbe830f74f4acda6583ce1c0caf` (Autor Nico Thomaier,
20.06.2026; das Override-Lexikon nach Angabe der Datei portiert aus kikiri-tts PR #28).

Lizenz: Apache License 2.0 (wie misaki, siehe https://github.com/hexgrad/misaki/blob/main/LICENSE).
Geändert: Importpfad von `EspeakG2P` und Ladeort von `de_overrides.json` (Einzelheiten im
Dateikopf). Der restliche Inhalt ist unverändert.

Laufzeit: `misaki.espeak.EspeakG2P` nutzt `phonemizer-fork` (GPL-3.0) und die über
`espeakng-loader` mitgelieferte eSpeak-NG-Bibliothek (GPL-3.0), wie schon die übrigen
espeak-Sprachen dieses Images.
