"""Löschfrist für die von /why/tts erzeugten Audiodateien.

Die Dateien sind vorgelesene KI-Antworten an Kinder und öffentlich über
<server>/output/<datei> abrufbar. Das Backend gibt die URL an die App weiter;
die App spielt sie sofort ab (Wiederholen-Knopf in derselben Sitzung). Ein
Aufräumjob im Container löscht sie nach OUTPUT_RETENTION_HOURS.

Gelöscht werden nur Dateien direkt im Ausgabeverzeichnis mit Audio-Endung.
Geloggt wird nur die Anzahl.
"""

import asyncio
import os
import time
from pathlib import Path

from loguru import logger

from ..core.config import settings

AUDIO_SUFFIXES = {".mp3", ".wav", ".opus", ".aac", ".flac", ".pcm"}
CLEANUP_INTERVAL_S = 15 * 60


def cleanup_output_dir(directory: str, max_age_hours: float, now: float | None = None) -> int:
    """Löscht Audiodateien älter als max_age_hours; gibt die Anzahl zurück."""
    if max_age_hours <= 0:
        return 0
    base = Path(directory)
    if not base.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - max_age_hours * 3600
    removed = 0
    for entry in base.iterdir():
        try:
            if (
                entry.is_file()
                and not entry.is_symlink()
                and entry.suffix.lower() in AUDIO_SUFFIXES
                and entry.stat().st_mtime < cutoff
            ):
                entry.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(f"Output cleanup: could not remove a file ({type(e).__name__})")
    return removed


async def run_output_cleanup_loop() -> None:
    """Läuft im Hintergrund, solange der Server läuft."""
    hours = settings.output_retention_hours
    logger.info(f"Output cleanup active: retention {hours}h, every {CLEANUP_INTERVAL_S // 60} min")
    while True:
        try:
            removed = await asyncio.to_thread(
                cleanup_output_dir, settings.why_output_dir, hours
            )
            if removed:
                logger.info(f"Output cleanup: removed {removed} file(s)")
        except Exception as e:  # nie den Server mitreißen
            logger.warning(f"Output cleanup failed: {type(e).__name__}")
        await asyncio.sleep(CLEANUP_INTERVAL_S)
