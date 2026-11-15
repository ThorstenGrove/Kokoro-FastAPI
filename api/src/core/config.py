import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Log-Level für loguru (Umgebungsvariable LOG_LEVEL). Standard INFO: Auf DEBUG
    # schreibt die kokoro-Bibliothek selbst Textausschnitte ins Log.
    log_level: str = "INFO"

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )

    # Ein Modell pro Stimme (VOICE_MODELS), z. B. für den deutschen Container:
    #   "dm_martin=/app/de/models/martin.pth,df_victoria=/app/de/models/victoria.pth"
    # Ist es gesetzt, wird das Standardmodell nicht geladen; jede Stimme braucht
    # dann einen Eintrag. Leer = bisheriges Verhalten (ein Modell für alle).
    voice_models: str | None = None
    # config.json für diese Modelle (Standard: <model_dir>/v1_0/config.json,
    # identisch mit hexgrad/Kokoro-82M)
    voice_model_config: str | None = None
    # Erlaubte lang_codes für /why/tts (ALLOWED_LANG_CODES, kommagetrennt, z. B. "d").
    # Andere werden mit HTTP 400 abgelehnt, damit das Backend auf den nächsten
    # Anbieter ausweicht. Leer = alle erlaubt (bisheriges Verhalten).
    allowed_lang_codes: str | None = None
    # Löschfrist für erzeugte Dateien in why_output_dir in Stunden
    # (OUTPUT_RETENTION_HOURS). 0 = nie löschen (bisheriges Verhalten).
    output_retention_hours: float = 0
    why_output_dir: str = "/output"

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    class Config:
        env_file = ".env"

    def voice_model_map(self) -> dict[str, str]:
        """VOICE_MODELS als {stimme: pfad}."""
        result: dict[str, str] = {}
        for item in (self.voice_models or "").split(","):
            if not item.strip():
                continue
            name, sep, path = item.partition("=")
            if not sep or not name.strip() or not path.strip():
                raise ValueError(f"Invalid VOICE_MODELS entry: {item!r}")
            result[name.strip()] = path.strip()
        return result

    def allowed_lang_code_set(self) -> set[str]:
        """ALLOWED_LANG_CODES als Menge (leer = alle erlaubt)."""
        return {
            code.strip().lower()
            for code in (self.allowed_lang_codes or "").split(",")
            if code.strip()
        }

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
