"""Why TTS router for file-based audio generation with timestamps."""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from ..inference.base import AudioChunk
from ..services.audio import AudioNormalizer, AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.tts_service import TTSService
from ..core.config import settings
from ..structures import CaptionedSpeechRequest, WordTimestamp
from .openai_compatible import process_and_validate_voices

router = APIRouter(tags=["why tts"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return await TTSService.create()


class WhyTTSResponse:
    """Response model for /why/tts endpoint"""
    
    def __init__(self, filename: str, timestamps: Optional[list[WordTimestamp]] = None, duration: Optional[float] = None):
        self.filename = filename
        self.timestamps = timestamps or []
        self.duration = duration
    
    def model_dump(self):
        """Convert to dictionary for JSON response"""
        response = {
            "filename": self.filename,
            "timestamps": [
                {
                    "word": ts.word,
                    "start_time": ts.start_time,
                    "end_time": ts.end_time
                }
                for ts in self.timestamps
            ]
        }
        if self.duration is not None:
            response["duration"] = self.duration
        return response


@router.post("/why/tts")
async def create_why_tts(
    request: CaptionedSpeechRequest,
    client_request: Request,
    tts_service: TTSService = Depends(get_tts_service),
):
    """Generate audio file with word-level timestamps and save to /output directory.
    
    This endpoint generates audio similar to captioned_speech but:
    - Does not support streaming (stream parameter is ignored)
    - Saves the audio file to /output directory with a random filename
    - Returns filename, timestamps, and duration (if available)
    
    Args:
        request: Request containing text and voice configuration
        
    Returns:
        JSON response with filename, timestamps, and duration
    """
    
    try:
        # Initialize TTS service and validate voice
        tts_service = await get_tts_service()
        voice_name = await process_and_validate_voices(request.voice, tts_service)

        # Instanz nur für bestimmte Sprachen (ALLOWED_LANG_CODES, z. B. der deutsche
        # Container): andere Sprachcodes mit 400 ablehnen, das Backend weicht dann
        # auf den nächsten Anbieter aus.
        allowed = settings.allowed_lang_code_set()
        if allowed:
            effective_lang = (request.lang_code or voice_name[:1]).lower()
            if effective_lang not in allowed:
                raise ValueError(
                    f"lang_code '{effective_lang}' is not served by this instance"
                )
        
        # Generate random filename
        random_filename = f"{uuid.uuid4().hex}.{request.response_format}"
        
        # Ensure output directory exists
        output_dir = Path("/output")
        output_dir.mkdir(exist_ok=True)
        
        # Full path for the output file
        output_path = output_dir / random_filename
        
        # Create streaming audio writer for the format
        writer = StreamingAudioWriter(request.response_format, sample_rate=24000)
        
        # Generate complete audio (non-streaming approach).
        # Datenschutz: Der Text ist eine KI-Antwort an ein Kind und wird nie geloggt,
        # ebenso wenig der Dateiname (er ist die öffentliche Audio-URL). Nur Länge,
        # Stimme, Sprachcode, Audiodauer und Rechenzeit.
        started = time.perf_counter()
        
        audio_data = await tts_service.generate_audio(
            text=request.input,
            voice=voice_name,
            writer=writer,
            speed=request.speed,
            return_timestamps=request.return_timestamps,
            volume_multiplier=request.volume_multiplier,
            normalization_options=request.normalization_options,
            lang_code=request.lang_code,
        )
        
        # Convert audio to requested format
        audio_data = await AudioService.convert_audio(
            audio_data,
            request.response_format,
            writer,
            is_last_chunk=False,
            trim_audio=False,
        )
        
        # Finalize audio conversion
        final = await AudioService.convert_audio(
            AudioChunk(np.array([], dtype=np.int16)),
            request.response_format,
            writer,
            is_last_chunk=True,
        )
        
        # Combine audio data
        complete_audio = audio_data.output + final.output
        
        # Calculate duration if audio data is available
        duration = None
        if audio_data.audio is not None and len(audio_data.audio) > 0:
            # Duration = number of samples / sample rate
            duration = len(audio_data.audio) / 24000.0
        
        # Write audio to file
        with open(output_path, "wb") as f:
            f.write(complete_audio)
        
        logger.info(
            f"why/tts ok: voice={voice_name} lang_code={request.lang_code} "
            f"chars={len(request.input)} audio_s={duration or 0:.2f} "
            f"took_ms={(time.perf_counter() - started) * 1000:.0f}"
        )
        
        # Clean up writer
        writer.close()
        
        # Create response
        response = WhyTTSResponse(
            filename=random_filename,
            timestamps=audio_data.word_timestamps,
            duration=duration
        )
        
        return JSONResponse(
            content=response.model_dump(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
            },
        )
        
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        
        try:
            writer.close()
        except:
            pass
        
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error",
            },
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {type(e).__name__}")
        
        try:
            writer.close()
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in why TTS generation: {type(e).__name__}")
        
        try:
            writer.close()
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
