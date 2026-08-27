import io
import wave
import asyncio
import os
import sys
from typing import AsyncGenerator
from loguru import logger
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

class ChatterboxTTSService(TTSService):
    """
    Custom Pipecat TTS Service for ResembleAI/chatterbox-turbo running on local CPU.
    Uses the native chatterbox-tts Python package for inference.
    """
    _MODEL_CACHE = {}
    _MODEL_LOAD_ERRORS = {}

    def __init__(self, model: str = "turbo", device: str = "cpu", audio_prompt_path: str = None, **kwargs):
        """
        Args:
            model: "turbo" for ChatterboxTurboTTS, "standard" for ChatterboxTTS, "multilingual" for ChatterboxMultilingualTTS
            device: "cpu" for CPU, "cuda" for GPU, "mps" for Apple Silicon
            audio_prompt_path: Optional path to reference audio for voice cloning (10s+ recommended)
        """
        super().__init__(**kwargs)
        self.model_type = model
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.tts_model = None
        self._load_error = None
        self.warmup_text = kwargs.pop("warmup_text", "Hi") if "warmup_text" in kwargs else "Hi"
        logger.info(f"Chatterbox {model} TTS initialized for {device}")

    def _cache_key(self):
        return (self.model_type, self.device)

    def _load_model(self):
        """Lazy-load the model on first use to avoid startup delays."""
        if self.tts_model is not None:
            return
        cache_key = self._cache_key()
        if cache_key in self.__class__._MODEL_CACHE:
            self.tts_model = self.__class__._MODEL_CACHE[cache_key]
            return
        if cache_key in self.__class__._MODEL_LOAD_ERRORS:
            self._load_error = self.__class__._MODEL_LOAD_ERRORS[cache_key]
            raise RuntimeError(f"Chatterbox model load previously failed: {self._load_error}")
        if self._load_error is not None:
            raise RuntimeError(f"Chatterbox model load previously failed: {self._load_error}")
        
        try:
            # Keep transformers on text-only paths when possible.
            os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
            self._disable_broken_torchvision()
            self._validate_audio_prompt_path()

            if self.model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info(f"Loading ChatterboxTurboTTS on {self.device}...")
                self.tts_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                logger.info("ChatterboxTurboTTS loaded successfully")
            elif self.model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logger.info(f"Loading ChatterboxMultilingualTTS on {self.device}...")
                self.tts_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                logger.info("ChatterboxMultilingualTTS loaded successfully")
            else:  # default to standard Chatterbox
                from chatterbox.tts import ChatterboxTTS
                logger.info(f"Loading ChatterboxTTS on {self.device}...")
                self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
                logger.info("ChatterboxTTS loaded successfully")

            self.__class__._MODEL_CACHE[cache_key] = self.tts_model
        except ImportError as e:
            self._load_error = e
            self.__class__._MODEL_LOAD_ERRORS[cache_key] = e
            logger.error(f"Failed to import Chatterbox: {e}. Run 'pip install chatterbox-tts'")
            raise
        except Exception as e:
            self._load_error = e
            self.__class__._MODEL_LOAD_ERRORS[cache_key] = e
            message = str(e)
            if (
                "torchvision::nms" in message
                or "torchvision' has no attribute 'extension" in message
                or "Failed to import transformers.models.llama.modeling_llama" in message
            ):
                logger.error(
                    "Chatterbox dependency mismatch detected (torch/torchvision/transformers). "
                    "Use matching torch + torchvision builds for your Python version, or uninstall torchvision if unused. "
                    "Then reinstall transformers and chatterbox-tts in the same environment."
                )
            logger.error(f"Failed to load Chatterbox model: {e}")
            raise

    def warmup(self):
        """Load model and run one tiny generation to reduce first-response latency."""
        self._load_model()
        try:
            self._generate_audio_tensor(self.warmup_text)
            logger.info("Chatterbox warmup completed")
        except Exception as e:
            logger.warning(f"Chatterbox warmup generation failed: {e}")

    def _validate_audio_prompt_path(self):
        if not self.audio_prompt_path:
            return

        if not os.path.exists(self.audio_prompt_path):
            logger.warning(
                f"audio_prompt_path does not exist: {self.audio_prompt_path}. Falling back to default voice."
            )
            return

        _, ext = os.path.splitext(self.audio_prompt_path)
        if ext.lower() != ".wav":
            logger.warning(
                f"Voice reference is '{ext}' format. WAV is recommended for best cloning speed and stability."
            )

        try:
            import torchaudio

            meta = torchaudio.info(self.audio_prompt_path)
            duration_secs = float(meta.num_frames) / float(meta.sample_rate) if meta.sample_rate else 0.0
            if duration_secs and duration_secs < 6.0:
                logger.warning(
                    f"Voice reference is short ({duration_secs:.1f}s). Use 6-12s for better cloning quality."
                )
        except Exception as e:
            logger.debug(f"Could not inspect audio_prompt_path metadata: {e}")

    def _generate_audio_tensor(self, text: str):
        kwargs = {"text": text}
        if self.audio_prompt_path and os.path.exists(self.audio_prompt_path):
            kwargs["audio_prompt_path"] = self.audio_prompt_path
        return self.tts_model.generate(**kwargs)

    def _to_audio_frame(self, wav):
        import numpy as np

        audio_np = wav.data.cpu().numpy() if hasattr(wav, "data") else wav.numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np[0]

        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        sample_rate = getattr(self.tts_model, "sr", 24000)

        return TTSAudioRawFrame(audio=audio_bytes, sample_rate=sample_rate, num_channels=1)

    def _to_audio_frame_fallback(self, wav):
        import tempfile
        import torchaudio as ta

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ta.save(tmp_path, wav, 24000)
            with wave.open(tmp_path, "rb") as wav_file:
                raw_pcm = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                return TTSAudioRawFrame(audio=raw_pcm, sample_rate=sample_rate, num_channels=1)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _disable_broken_torchvision(self):
        """
        If torchvision is installed but broken for the current torch build,
        force transformers to treat torchvision as unavailable.
        """
        try:
            import torchvision  # noqa: F401
            return
        except Exception as e:
            logger.warning(
                f"Torchvision import failed ({e}). Disabling torchvision-dependent transformers paths."
            )

        # Remove any partially initialized module to avoid circular import leftovers.
        sys.modules.pop("torchvision", None)

        # Patch transformers import utils so is_torchvision_available() returns False.
        try:
            from transformers.utils import import_utils as tf_import_utils

            if hasattr(tf_import_utils, "_torchvision_available"):
                tf_import_utils._torchvision_available = False
            if hasattr(tf_import_utils, "_torchvision_version"):
                tf_import_utils._torchvision_version = "0"
        except Exception as patch_err:
            logger.debug(f"Unable to patch transformers torchvision flags: {patch_err}")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Chatterbox TTS generating audio for: {text[:100]}...")
        yield TTSStartedFrame()
        
        try:
            # Load model on first call (lazy initialization)
            self._load_model()

            wav = await asyncio.to_thread(self._generate_audio_tensor, text)
            try:
                frame = self._to_audio_frame(wav)
                logger.debug(f"Generated {len(frame.audio)} bytes at {frame.sample_rate}Hz")
                yield frame
            except Exception as e_convert:
                logger.error(f"Failed to convert audio tensor: {e_convert}")
                try:
                    yield self._to_audio_frame_fallback(wav)
                except Exception as e_fallback:
                    logger.error(f"Fallback conversion also failed: {e_fallback}")
                    raise

        except Exception as e:
            logger.error(f"Chatterbox TTS generation failed: {e}")
            
        yield TTSStoppedFrame()
