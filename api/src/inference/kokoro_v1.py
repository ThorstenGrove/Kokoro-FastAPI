"""Clean Kokoro implementation with controlled resource management."""

import os
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import model_config
from ..structures.schemas import WordTimestamp
from .base import AudioChunk, BaseModelBackend


GERMAN_LANG_CODE = "d"


def _create_german_pipeline(device: str) -> KPipeline:
    """KPipeline für Deutsch (lang_code "d").

    kokoro 0.9.2 kennt "d" nicht (LANG_CODES). Die Pipeline wird deshalb als
    espeak-Pipeline ohne Modell angelegt und danach auf den deutschen G2P
    umgestellt. Für alles außer "a"/"b" ruft KPipeline nur self.g2p(chunk) auf
    und erwartet (phoneme, None); genau das liefert DEG2P. Das Modell wird pro
    Aufruf übergeben (ein Modell pro Stimme, siehe _model_for_voice).
    """
    from ..services.text_processing.german_g2p import DEG2P

    pipeline = KPipeline(
        lang_code="e", repo_id="hexgrad/Kokoro-82M", model=False, device=device
    )
    pipeline.lang_code = GERMAN_LANG_CODE
    pipeline.g2p = DEG2P()
    return pipeline


def _load_checkpoint_checked(model: KModel, path: str) -> None:
    """Lädt einen Kokoro-Checkpoint vollständig und bricht bei Lücken ab.

    Die kikiri-Checkpoints sind mit der neuen Weight-Norm-API von PyTorch
    gespeichert ("<x>.parametrizations.weight.original0/1"), kokoro 0.9.2 nutzt
    die alte ("<x>.weight_g/weight_v"). KModel lädt solche Schlüssel nicht und
    lässt die Gewichte zufällig. Hier werden sie umbenannt.

    Fehlen dürfen nur die affinen Parameter der InstanceNorm in AdaIN
    ("*.norm.weight/bias"): StyleTTS2 trainiert sie nicht (affine=False), die
    Startwerte 1/0 entsprechen genau dem.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing_total = 0
    for key, sub in state.items():
        fixed = {}
        for name, value in sub.items():
            if name.startswith("module."):
                name = name[len("module."):]
            name = name.replace(
                ".parametrizations.weight.original0", ".weight_g"
            ).replace(".parametrizations.weight.original1", ".weight_v")
            fixed[name] = value
        result = getattr(model, key).load_state_dict(fixed, strict=False)
        bad_missing = [
            k for k in result.missing_keys
            if not (k.endswith(".norm.weight") or k.endswith(".norm.bias"))
        ]
        if bad_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint {os.path.basename(path)}: '{key}' missing "
                f"{bad_missing[:3]} unexpected {result.unexpected_keys[:3]}"
            )
        missing_total += len(result.missing_keys)
    logger.info(
        f"Checkpoint {os.path.basename(path)} loaded completely "
        f"({missing_total} untrained AdaIN norm params left at 1/0)"
    )


class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._model: Optional[KModel] = None
        self._pipelines: Dict[str, KPipeline] = {}  # Store pipelines by lang_code
        # Ein Modell pro Stimme (VOICE_MODELS), z. B. die deutschen kikiri-Checkpoints,
        # deren Voicepacks nur zu ihrem eigenen Feintuning passen.
        self._voice_models: Dict[str, KModel] = {}

    def _to_device(self, model: KModel) -> KModel:
        """Modell auf das konfigurierte Gerät legen."""
        if self._device == "mps":
            return model.to(torch.device("mps"))
        if self._device == "cuda":
            return model.cuda()
        return model.cpu()

    def _load_voice_models(self, voice_models: Dict[str, str]) -> None:
        """Lädt je Stimme einen eigenen Checkpoint (Kokoro-Format, weights_only)."""
        config_path = settings.voice_model_config or os.path.join(
            settings.model_dir, "v1_0", "config.json"
        )
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}")
        for voice_name, model_path in voice_models.items():
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Voice model not found: {model_path}")
            logger.info(f"Loading voice model for {voice_name} on {self._device}")
            # KModel lädt mit torch.load(..., weights_only=True), schluckt aber
            # abweichende Schlüssel stillschweigend (strict=False). Deshalb danach
            # vollständig und geprüft nachladen.
            model = KModel(
                repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path
            )
            _load_checkpoint_checked(model, model_path)
            self._voice_models[voice_name] = self._to_device(model.eval())
        # Erstes Modell als Standard, damit is_loaded und Aufrufer ohne Stimme funktionieren
        self._model = next(iter(self._voice_models.values()))

    def _model_for_voice(self, voice_name: str) -> KModel:
        """Modell zur Stimme; ohne VOICE_MODELS immer das eine Standardmodell."""
        if not self._voice_models:
            return self._model
        model = self._voice_models.get(voice_name)
        if model is None:
            # Ein Voicepack mit fremdem Modell klingt falsch; lieber ablehnen.
            raise RuntimeError(f"No model configured for voice '{voice_name}'")
        return model

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        voice_models = settings.voice_model_map()
        if voice_models:
            # Modus "ein Modell pro Stimme": das Standardmodell wird nicht geladen.
            try:
                self._load_voice_models(voice_models)
                return
            except FileNotFoundError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to load voice models: {e}")

        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")

            # Load model and let KModel handle device mapping
            self._model = KModel(config=config_path, model=model_path).eval()
            # For MPS, manually move ISTFT layers to CPU while keeping rest on MPS
            if self._device == "mps":
                logger.info(
                    "Moving model to MPS device with CPU fallback for unsupported operations"
                )
                self._model = self._model.to(torch.device("mps"))
            elif self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language code.

        Args:
            lang_code: Language code to use

        Returns:
            KPipeline instance for the language
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if lang_code not in self._pipelines:
            logger.info(f"Creating new pipeline for language code: {lang_code}")
            if lang_code == GERMAN_LANG_CODE:
                self._pipelines[lang_code] = _create_german_pipeline(self._device)
            else:
                self._pipelines[lang_code] = KPipeline(
                    lang_code=lang_code, model=self._model, device=self._device
                )
        return self._pipelines[lang_code]

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice
                voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            # Load voice tensor with proper device mapping
            voice_tensor = await paths.load_voice_tensor(
                voice_path, device=self._device
            )
            # Save back to a temporary file with proper device mapping
            import tempfile

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
            )
            await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Use provided lang_code, settings voice code override, or first letter of voice name
            if lang_code:  # api is given priority
                pipeline_lang_code = lang_code
            elif settings.default_voice_code:  # settings is next priority
                pipeline_lang_code = settings.default_voice_code
            else:  # voice name is default/fallback
                pipeline_lang_code = voice_name[0].lower()

            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio from tokens with lang_code '{pipeline_lang_code}' ({len(tokens)} chars)"
            )
            for result in pipeline.generate_from_tokens(
                tokens=tokens,
                voice=voice_path,
                speed=speed,
                model=self._model_for_voice(voice_name),
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate_from_tokens(
                    tokens, voice, speed, lang_code
                ):
                    yield chunk
            raise

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            voice_name: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice
                voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            # Load voice tensor with proper device mapping
            voice_tensor = await paths.load_voice_tensor(
                voice_path, device=self._device
            )
            # Save back to a temporary file with proper device mapping
            import tempfile

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
            )
            await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Use provided lang_code, settings voice code override, or first letter of voice name
            pipeline_lang_code = (
                lang_code
                if lang_code
                else (
                    settings.default_voice_code
                    if settings.default_voice_code
                    else voice_name[0].lower()
                )
            )
            pipeline = self._get_pipeline(pipeline_lang_code)

            logger.debug(
                f"Generating audio for text with lang_code '{pipeline_lang_code}' ({len(text)} chars)"
            )
            for result in pipeline(
                text,
                voice=voice_path,
                speed=speed,
                model=self._model_for_voice(voice_name),
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    word_timestamps = None
                    if (
                        return_timestamps
                        and hasattr(result, "tokens")
                        and result.tokens
                    ):
                        word_timestamps = []
                        current_offset = 0.0
                        logger.debug(
                            f"Processing chunk timestamps with {len(result.tokens)} tokens"
                        )
                        if result.pred_dur is not None:
                            try:
                                # Add timestamps with offset
                                for token in result.tokens:
                                    if not all(
                                        hasattr(token, attr)
                                        for attr in [
                                            "text",
                                            "start_ts",
                                            "end_ts",
                                        ]
                                    ):
                                        continue
                                    if not token.text or not token.text.strip():
                                        continue

                                    start_time = float(token.start_ts) + current_offset
                                    end_time = float(token.end_ts) + current_offset
                                    word_timestamps.append(
                                        WordTimestamp(
                                            word=str(token.text).strip(),
                                            start_time=start_time,
                                            end_time=end_time,
                                        )
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Failed to process timestamps for chunk: {e}"
                                )

                    yield AudioChunk(
                        result.audio.numpy(), word_timestamps=word_timestamps
                    )
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate(text, voice, speed, lang_code):
                    yield chunk
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > model_config.pytorch_gpu.memory_threshold
        # MPS doesn't provide memory management APIs
        return False

    def _clear_memory(self) -> None:
        """Clear device memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self._device == "mps":
            # Empty cache if available (future-proofing)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def unload(self) -> None:
        """Unload model and free resources."""
        self._voice_models.clear()
        if self._model is not None:
            del self._model
            self._model = None
        for pipeline in self._pipelines.values():
            del pipeline
        self._pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
