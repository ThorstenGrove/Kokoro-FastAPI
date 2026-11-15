# Deutscher Kokoro-Container (nur lang_code "d", Stimmen dm_martin und df_victoria).
#
# Basis wie das Produktiv-Image (docker/gpu): CUDA 12.4 + torch 2.4.1+cu124.
# Begründung: Diese Kombination läuft auf ki.home nachweislich auf der Tesla V100
# (sm_70 in torch.cuda.get_arch_list()). Die kikiri-Checkpoints sind normale
# Kokoro-82M-Gewichte und brauchen kein neueres torch. cu126 würde auch gehen
# (Volta bis torch 2.14), bringt hier aber nichts und wäre ein ungetesteter Stack;
# cu128 ab 2.11 und CUDA 13 können Volta nicht.
#
# Bauen (aus dem Repo-Wurzelverzeichnis über docker/gpu-de/docker-compose.yml):
#   docker compose -f docker/gpu-de/docker-compose.yml build
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv in fester Version (statt Installationsskript ohne Version)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3.10 python3-venv espeak-ng espeak-ng-data \
        git libsndfile1 curl ca-certificates ffmpeg g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/de/models /app/de/voices && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Modelle aus festen Hugging-Face-Revisionen (Commit-Hash) mit SHA-256-Prüfung.
# Eigene Schicht vor dem Code, damit Code-Änderungen nicht neu herunterladen.
#   kikiri-tts/kikiri-german-martin   @ 1e9dcd16ed48fda0a7a1f62e5e37130a5fdf10d9 (22.04.2026)
#   kikiri-tts/kikiri-german-victoria @ ce81e200ff9203e1a3b042cd678c48e3ffb85cef (22.04.2026)
# Lizenz laut Modellkarten: Apache-2.0. Format: PyTorch-State-Dict (.pth), wird von
# KModel mit torch.load(..., weights_only=True) geladen; Voicepacks ebenso.
ARG HF=https://huggingface.co/kikiri-tts
ARG MARTIN_REV=1e9dcd16ed48fda0a7a1f62e5e37130a5fdf10d9
ARG VICTORIA_REV=ce81e200ff9203e1a3b042cd678c48e3ffb85cef
RUN cd /app/de && \
    curl -fsSL --retry 3 -o models/martin.pth   "$HF/kikiri-german-martin/resolve/$MARTIN_REV/kikiri_german_martin_ep10.pth" && \
    curl -fsSL --retry 3 -o voices/dm_martin.pt "$HF/kikiri-german-martin/resolve/$MARTIN_REV/voices/martin.pt" && \
    curl -fsSL --retry 3 -o models/victoria.pth   "$HF/kikiri-german-victoria/resolve/$VICTORIA_REV/kikiri_german_victoria_ep10.pth" && \
    curl -fsSL --retry 3 -o voices/df_victoria.pt "$HF/kikiri-german-victoria/resolve/$VICTORIA_REV/voices/victoria.pt" && \
    printf '%s\n' \
      "f194905cc97d4ef16e0f2613c2d277203e0c0f0f533e0a468c7ec76c566259dc  models/martin.pth" \
      "dc2e245e631ceaa9d30b327dac076342649c62f953021bcccafbe254ee43e553  voices/dm_martin.pt" \
      "128d1c4a4184a459f23cb307a6cbbb8e9c076c2526f86e18594c1b320a81e00c  models/victoria.pth" \
      "18dbcf0afa7b737e450e6ba769ee898a5aae1d90218265ff4c95422f4693da50  voices/df_victoria.pt" \
      > SHA256SUMS && \
    sha256sum -c SHA256SUMS

# Abhängigkeiten wie im Produktiv-Container, Versionen fest (constraints.txt).
# torch kommt über [tool.uv.sources] in pyproject.toml aus dem cu124-Index.
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml
COPY --chown=appuser:appuser docker/gpu-de/constraints.txt ./constraints.txt
RUN uv venv --python 3.10 && \
    uv pip install --no-cache -r pyproject.toml --extra gpu -c constraints.txt

# Code ins Image (kein Bind-Mount wie beim Produktiv-Container, damit ein
# Branch-Wechsel im Checkout den laufenden Container nicht verändert).
COPY --chown=appuser:appuser api ./api

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    HF_HUB_OFFLINE=1 \
    VOICE_MODELS="dm_martin=/app/de/models/martin.pth,df_victoria=/app/de/models/victoria.pth" \
    VOICE_MODEL_CONFIG=/app/api/src/models/v1_0/config.json \
    VOICES_DIR=/app/de/voices \
    DEFAULT_VOICE=df_victoria \
    ALLOWED_LANG_CODES=d \
    WHY_OUTPUT_DIR=/output \
    OUTPUT_RETENTION_HOURS=24 \
    ENABLE_WEB_PLAYER=false \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8880/health || exit 1

# Ohne Zugriffslog: der Server schreibt pro Anfrage selbst eine Zeile ohne Inhalte.
CMD ["python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "info", "--no-access-log"]
