"""
Cudara Inference Server (Transformers Native).

Lightweight CUDA inference server with Ollama-compatible API.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import gc
import io
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Union

import torch
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import login, snapshot_download
from PIL import ExifTags, UnidentifiedImageError
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TextIteratorStreamer,
    pipeline,
)

# ---------------------------------------------------------------------------
# Version — single source of truth
# ---------------------------------------------------------------------------
VERSION = "0.0.1"

# ---------------------------------------------------------------------------
# Patch older transformers builds missing PytorchGELUTanh
# ---------------------------------------------------------------------------
try:
    import transformers.activations

    if not hasattr(transformers.activations, "PytorchGELUTanh"):
        transformers.activations.PytorchGELUTanh = type("PytorchGELUTanh", (object,), {})  # type: ignore
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq.*")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cudara")

# ---------------------------------------------------------------------------
# CUDA setup
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = torch.cuda.is_available()
if _CUDA_AVAILABLE:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if HF_TOKEN := os.getenv("HF_TOKEN"):
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as exc:
        logger.warning("HF_TOKEN login failed: %s", exc)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.getenv("CUDARA_MODELS_DIR", "models"))
TEMP_DIR = Path(os.getenv("CUDARA_TEMP_DIR", "temp_uploads"))
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

GPU_LOCK = asyncio.Lock()
IDLE_TIMEOUT_SECONDS = int(os.getenv("CUDARA_IDLE_TIMEOUT", "300"))
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("CUDARA_MAX_TOKENS", "512"))
STREAMER_TIMEOUT = int(os.getenv("CUDARA_STREAMER_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Image processing limits
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = int(os.getenv("CUDARA_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))  # 20 MB
MAX_IMAGE_PIXELS = int(os.getenv("CUDARA_MAX_IMAGE_PIXELS", str(30_000_000)))  # ~30 MP
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}


# ===== Lazy imports for heavy optional deps =====
def _load_qwen_vl_utils():
    """Lazily import qwen_vl_utils only when a VLM is actually used."""
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        return process_vision_info
    except ImportError as exc:
        raise AppError(
            "qwen-vl-utils is required for vision models. Install with: pip install qwen-vl-utils",
            500,
            "missing_dependency",
        ) from exc


def _load_qwen_vl_model_class():
    """Lazily import Qwen VL model class."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise AppError(
            "transformers version does not support Qwen2.5-VL. Upgrade transformers.",
            500,
            "missing_dependency",
        ) from exc


# ===================================================================
# Error types
# ===================================================================
class AppError(Exception):
    """Custom application error mapped to standard JSON error format."""

    def __init__(self, message: str, status_code: int = 500, code: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


# ===================================================================
# Pydantic models — requests
# ===================================================================
class ModelOptions(BaseModel):
    """Runtime options that control text generation."""

    seed: Optional[int] = Field(None, description="Random seed for reproducible outputs")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Randomness in generation")
    top_k: Optional[int] = Field(None, ge=1, description="Limits next token selection to K most likely")
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, description="Cumulative probability threshold")
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum probability threshold")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    num_ctx: Optional[int] = Field(None, ge=1, description="Context length size")
    num_predict: Optional[int] = Field(None, ge=1, le=32768, description="Maximum number of tokens to generate")
    model_config = ConfigDict(extra="allow")


class GenerateRequest(BaseModel):
    """Payload for generating a response from a model."""

    model: str = Field(..., min_length=1, description="Model name")
    prompt: Optional[str] = Field(None, description="Text prompt")
    suffix: Optional[str] = Field(None, description="Text after user prompt for FIM")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Output format constraints")
    system: Optional[str] = Field(None, description="System prompt")
    stream: bool = Field(True, description="Stream partial responses")
    think: Optional[Union[bool, Literal["high", "medium", "low"]]] = Field(None, description="Thinking output")
    raw: bool = Field(False, description="Raw response without template")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Model keep-alive duration")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")
    logprobs: Optional[bool] = Field(None, description="Return log probabilities")
    top_logprobs: Optional[int] = Field(None, ge=1, le=20, description="Number of top logprobs")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate base64 images are decodable and non-empty.

        Note: This validator intentionally does NOT verify the bytes are
        a valid image because the `images` field is overloaded — it also
        carries audio data for ASR models (via the `is_audio` option).
        Full image validation happens downstream in _safe_decode_base64_image,
        which only runs on the VLM image-to-text path.
        """
        if v is None:
            return v
        for i, img in enumerate(v):
            try:
                raw = base64.b64decode(img, validate=True)
            except Exception:
                raise ValueError(f"images[{i}] is not valid base64")
            if len(raw) == 0:
                raise ValueError(f"images[{i}] is empty")
        return v


class ChatMessage(BaseModel):
    """A single message in the chat history."""

    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Message author")
    content: str = Field(..., description="Message text content")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")
    tool_calls: Optional[List[Any]] = Field(None, description="Tool calls")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate base64 images are decodable.

        FIX: The original ChatMessage had NO validation. Malformed
        base64 through /api/chat crashed inside the processor.
        """
        if v is None:
            return v
        for i, img in enumerate(v):
            try:
                raw = base64.b64decode(img, validate=True)
            except Exception:
                raise ValueError(f"images[{i}] is not valid base64")
            if len(raw) == 0:
                raise ValueError(f"images[{i}] is empty")
            try:
                with PILImage.open(io.BytesIO(raw)) as test_img:
                    test_img.verify()
            except Exception:
                raise ValueError(f"images[{i}] is valid base64 but not a valid image file")
        return v


class ChatRequest(BaseModel):
    """Payload for generating a chat response."""

    model: str = Field(..., min_length=1, description="Model name")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Chat history")
    tools: Optional[List[Any]] = Field(None, description="Available functions")
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = Field(None, description="Response format")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")
    stream: bool = Field(True, description="Stream responses")
    think: Optional[Union[bool, Literal["high", "medium", "low"]]] = Field(None, description="Include thinking")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Keep-alive duration")
    logprobs: Optional[bool] = Field(None, description="Return log probabilities")
    top_logprobs: Optional[int] = Field(None, ge=1, le=20, description="Number of top logprobs")


class EmbedRequest(BaseModel):
    """Payload for generating embeddings."""

    model: str = Field(..., min_length=1, description="Model name")
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    truncate: bool = Field(True, description="Truncate inputs exceeding context window")
    dimensions: Optional[int] = Field(None, ge=1, description="Output dimensions")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Keep-alive duration")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        """Ensure input is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("input must not be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("input list must not be empty")
        return v


class ShowRequest(BaseModel):
    """Payload for showing model details."""

    model: str = Field(..., min_length=1, description="Model name")
    verbose: bool = Field(False, description="Include verbose fields")


class PullRequest(BaseModel):
    """Payload for pulling a model."""

    model: str = Field(..., min_length=1, description="Model to download")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(True, description="Stream progress")


class DeleteRequest(BaseModel):
    """Payload for deleting a model."""

    model: str = Field(..., min_length=1, description="Model to delete")


# ===================================================================
# Pydantic models — responses
# ===================================================================
class TokenLogprob(BaseModel):
    """Log probability for a single token alternative."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class Logprob(BaseModel):
    """Log probability for a generated token."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[TokenLogprob]] = None


class ModelDetails(BaseModel):
    """Model format and architecture metadata."""

    format: str
    family: str
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class ModelSummary(BaseModel):
    """Summary for a locally available model."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class ListResponse(BaseModel):
    """List of available models."""

    models: List[ModelSummary]


class Ps(BaseModel):
    """Running model information."""

    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: str
    size_vram: int
    context_length: int = 0


class PsResponse(BaseModel):
    """Running models response."""

    models: List[Ps]


class ShowResponse(BaseModel):
    """Detailed model information."""

    parameters: Optional[str] = None
    license: Optional[str] = None
    modified_at: str
    details: ModelDetails
    template: Optional[str] = None
    capabilities: List[str]
    model_info: Dict[str, Any]


class EmbedResponse(BaseModel):
    """Embedding or rerank response."""

    model: str
    embeddings: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


# ===================================================================
# Internal models
# ===================================================================
class ModelStatus(str, Enum):
    """Model installation state."""

    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelConfig(BaseModel):
    """Model configuration from models.json."""

    description: Optional[str] = None
    task: str = "text-generation"
    backend: str = "transformers"
    system_prompt: Optional[str] = None


class RegistryItem(BaseModel):
    """Model installation tracking."""

    status: ModelStatus
    local_path: Optional[str] = None
    error_message: Optional[str] = None


# ===================================================================
# Utilities
# ===================================================================
def _utcnow_iso() -> str:
    """Return current UTC time as ISO 8601."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_digest(model_id: str) -> str:
    """Deterministic digest from model id."""
    return "sha256:" + base64.b16encode(model_id.encode()).decode().lower()[:64]


def _dir_size(path: Path) -> int:
    """Total size of all files under path in bytes."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _parse_keep_alive(value: Union[str, int, None]) -> Optional[float]:
    """Parse keep_alive into seconds. Returns None if not set, -1 for infinite, 0 for immediate unload."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).lower().strip()
    if value in ("0", "0s", "0m"):
        return 0.0
    if value == "-1":
        return -1.0

    multipliers = {"h": 3600.0, "m": 60.0, "s": 1.0}
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            try:
                return float(value[:-1]) * mult
            except ValueError:
                return float(IDLE_TIMEOUT_SECONDS)

    try:
        return float(value)
    except ValueError:
        return float(IDLE_TIMEOUT_SECONDS)


def inject_json_instruction(messages: List[Dict[str, Any]], format_req: Union[str, Dict[str, Any], None]) -> bool:
    """Inject JSON constraints into system prompts if strict formatting is requested."""
    if not format_req:
        return False

    if isinstance(format_req, dict):
        schema_str = json.dumps(format_req, indent=2)
        instruction = (
            f"You are a strict data-generating API. Respond ONLY with valid JSON "
            f"matching this schema:\n{schema_str}\n"
            "No explanations, no markdown, no conversational text."
        )
    else:
        instruction = (
            "You are a strict data-generating API. Respond ONLY with valid, parseable JSON. "
            "No explanations, no markdown, no conversational text."
        )

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] += f"\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})

    return True


def _extract_options(
    options: Optional[ModelOptions], logprobs: Optional[bool] = None, top_logprobs: Optional[int] = None
) -> Dict[str, Any]:
    """Consolidate inference options."""
    opts = options.model_dump(exclude_unset=True) if options else {}
    if logprobs:
        opts["logprobs"] = True
        if top_logprobs:
            opts["top_logprobs"] = top_logprobs
    return opts


# ===================================================================
# Image processing utilities
# ===================================================================
def _detect_mime_type(raw_bytes: bytes) -> str:
    """Detect actual image MIME type from magic bytes.

    This fixes the root cause of the WhatsApp crash: WebP images
    were being labeled as image/jpeg, causing decoder failures.
    """
    signatures = [
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"RIFF", None),  # WebP needs secondary check
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"BM", "image/bmp"),
        (b"II\x2a\x00", "image/tiff"),
        (b"MM\x00\x2a", "image/tiff"),
    ]

    for magic, mime in signatures:
        if raw_bytes[: len(magic)] == magic:
            if mime is not None:
                return mime
            # RIFF container — check for WebP
            if len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WEBP":
                return "image/webp"
            break

    # Fallback: let Pillow identify it
    try:
        with PILImage.open(io.BytesIO(raw_bytes)) as img:
            fmt = img.format
            if fmt:
                return f"image/{fmt.lower()}"
    except Exception:
        pass

    return "image/jpeg"


def _validate_and_normalize_image(
    raw_bytes: bytes,
    image_index: int = 0,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[bytes, str]:
    """Validate, normalize, and return (cleaned_bytes, mime_type).

    Addresses:
    - Corrupt/truncated images crashing the processor
    - RGBA images causing wrong tensor shapes
    - EXIF rotation causing sideways text (destroys OCR accuracy)
    - Oversized images causing CPU OOM before VRAM guard fires
    - Non-image files disguised as base64 images
    """
    # --- Size gate (before any decoding to prevent CPU OOM) ---
    if len(raw_bytes) > max_bytes:
        raise AppError(
            f"Image [{image_index}] exceeds maximum size "
            f"({len(raw_bytes) / 1024 / 1024:.1f} MB > {max_bytes / 1024 / 1024:.0f} MB limit).",
            400,
            "image_too_large",
        )

    # --- Decode with Pillow (catches corrupt/truncated/non-image) ---
    try:
        img = PILImage.open(io.BytesIO(raw_bytes))
        img.load()  # Force full decode — catches truncated files
    except UnidentifiedImageError:
        raise AppError(
            f"Image [{image_index}] is not a recognized image format. Supported: JPEG, PNG, WebP, GIF, BMP, TIFF.",
            400,
            "invalid_image",
        )
    except (OSError, Exception) as exc:
        raise AppError(
            f"Image [{image_index}] is corrupt or truncated: {type(exc).__name__}: {exc}",
            400,
            "corrupt_image",
        )

    # --- Format check ---
    if img.format and img.format.upper() not in SUPPORTED_IMAGE_FORMATS:
        raise AppError(
            f"Image [{image_index}] format '{img.format}' is not supported.",
            400,
            "unsupported_format",
        )

    # --- Pixel count gate (downscale instead of reject) ---
    result: PILImage.Image = img
    w, h = result.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(
            "Image [%d] downscaled from %dx%d to %dx%d (pixel limit).",
            image_index,
            w,
            h,
            new_w,
            new_h,
        )
        result = result.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

    # --- EXIF orientation fix ---
    # Without this, text in rotated phone/WhatsApp images appears
    # sideways to the model, destroying text recognition accuracy.
    try:
        exif = result.getexif()
        orientation_key = None
        for tag_id, tag_name in ExifTags.TAGS.items():
            if tag_name == "Orientation":
                orientation_key = tag_id
                break

        if orientation_key and orientation_key in exif:
            orientation = exif[orientation_key]
            _T = PILImage.Transpose
            transforms: Dict[int, List[PILImage.Transpose]] = {
                2: [_T.FLIP_LEFT_RIGHT],
                3: [_T.ROTATE_180],
                4: [_T.FLIP_TOP_BOTTOM],
                5: [_T.FLIP_LEFT_RIGHT, _T.ROTATE_90],
                6: [_T.ROTATE_270],
                7: [_T.FLIP_LEFT_RIGHT, _T.ROTATE_270],
                8: [_T.ROTATE_90],
            }
            for op in transforms.get(orientation, []):
                result = result.transpose(op)
    except (AttributeError, KeyError, Exception):
        pass  # No EXIF or no orientation — fine

    # --- Ensure RGB (models expect 3-channel input) ---
    if result.mode == "RGBA":
        background = PILImage.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])
        result = background
    elif result.mode == "P":
        # Palette mode — convert via RGBA to handle transparency
        result = result.convert("RGBA")
        background = PILImage.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])
        result = background
    elif result.mode == "LA":
        # Grayscale with alpha
        result = result.convert("RGBA")
        background = PILImage.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])
        result = background
    elif result.mode not in ("RGB",):
        result = result.convert("RGB")

    # --- Re-encode to PNG (lossless, avoids JPEG recompression artifacts) ---
    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=False)
    clean_bytes = buf.getvalue()

    return clean_bytes, "image/png"


def _safe_decode_base64_image(b64_string: str, image_index: int = 0) -> tuple[bytes, str]:
    """Decode base64 string, validate, and normalize the image.

    Returns:
        Tuple of (normalized PNG bytes, MIME type "image/png")
    """
    try:
        raw_bytes = base64.b64decode(b64_string, validate=True)
    except Exception:
        raise AppError(
            f"Image [{image_index}] contains invalid base64 encoding.",
            400,
            "invalid_base64",
        )

    if len(raw_bytes) == 0:
        raise AppError(
            f"Image [{image_index}] is empty (0 bytes after base64 decode).",
            400,
            "empty_image",
        )

    return _validate_and_normalize_image(raw_bytes, image_index)


# ===================================================================
# Model Manager
# ===================================================================
class ModelManager:
    """Thread-safe model registry and download manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registry_path = Path("registry.json")
        if not self._registry_path.exists():
            self._save_json(self._registry_path, {})

    def _save_json(self, path: Path, data: Any) -> None:
        """Atomically write JSON to disk via temp file rename."""
        tmp = path.with_suffix(".tmp")
        try:
            with self._lock:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(path)  # atomic on POSIX
        except OSError as exc:
            logger.error("Failed to write %s: %s", path, exc)
            tmp.unlink(missing_ok=True)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON safely."""
        with self._lock:
            if not path.exists():
                return {}
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to read %s: %s", path, exc)
                return {}

    def get_allowed(self) -> Dict[str, ModelConfig]:
        """Retrieve allowed models from models.json."""
        data = self._load_json(Path("models.json"))
        result = {}
        for k, v in data.items():
            if k.startswith("_"):
                continue
            try:
                result[k] = ModelConfig(**v)
            except Exception as exc:
                logger.warning("Invalid config for model %s: %s", k, exc)
        return result

    def get_registry(self) -> Dict[str, RegistryItem]:
        """Retrieve downloaded model registry."""
        data = self._load_json(self._registry_path)
        result = {}
        for k, v in data.items():
            try:
                result[k] = RegistryItem(**v)
            except Exception as exc:
                logger.warning("Invalid registry entry %s: %s", k, exc)
        return result

    def update_registry(self, model_id: str, **kwargs: Any) -> None:
        """Update a model entry in the registry."""
        reg = self._load_json(self._registry_path)
        reg.setdefault(model_id, {}).update(kwargs)
        self._save_json(self._registry_path, reg)

    def download_task(self, model_id: str) -> None:
        """Download model weights from Hugging Face."""
        try:
            local_dir = MODELS_DIR / model_id.replace("/", "--")
            self.update_registry(model_id, status=ModelStatus.DOWNLOADING)
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.ot", "*.onnx"],
            )
            self.update_registry(model_id, status=ModelStatus.READY, local_path=str(local_dir))
            logger.info("Model %s downloaded successfully.", model_id)
        except Exception as exc:
            self.update_registry(model_id, status=ModelStatus.ERROR, error_message=str(exc))
            logger.error("Failed to download %s: %s", model_id, exc)

    def delete_model(self, model_id: str) -> None:
        """Remove model assets and registry entry."""
        reg = self._load_json(self._registry_path)
        entry = reg.pop(model_id, None)
        if entry is None:
            raise AppError(f"Model '{model_id}' not found in registry.", 404)

        path = entry.get("local_path")
        if path:
            p = Path(path)
            if p.exists():
                try:
                    shutil.rmtree(p)
                except OSError as exc:
                    logger.error("Failed to remove %s: %s", p, exc)
                    raise AppError(f"Failed to delete model files: {exc}", 500)

        self._save_json(self._registry_path, reg)


# ===================================================================
# Logits Processor
# ===================================================================
class NaNSanitizerProcessor(LogitsProcessor):
    """Sanitize NaN/Inf logits caused by FP16 overflow."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Replace NaN/Inf with dtype-safe finite values."""
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            info = torch.finfo(scores.dtype)
            scores.nan_to_num_(nan=info.min, posinf=info.max, neginf=info.min)
        return scores


# ===================================================================
# Inference Engine
# ===================================================================
class InferenceEngine:
    """Manages model lifecycle and PyTorch inference."""

    def __init__(self, mgr: ModelManager) -> None:
        self.manager = mgr
        self.active_id: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.processor: Optional[Any] = None
        self._lock = threading.RLock()
        self.last_interaction: float = 0.0
        self.active_requests: int = 0
        self.keep_alive_timeout: float = float(IDLE_TIMEOUT_SECONDS)

    @contextlib.contextmanager
    def request_context(self) -> Iterator[None]:
        """Track active GPU usage to prevent idle unloading mid-request."""
        with self._lock:
            self.active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self.active_requests -= 1
                self.last_interaction = time.time()

    def unload(self) -> None:
        """Release VRAM."""
        with self._lock:
            if self.model is None:
                return
            model_id = self.active_id
            logger.info("Unloading model %s from VRAM.", model_id)
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.active_id = None

        # GC outside the lock — no need to hold it during collection
        gc.collect()
        gc.collect()
        if _CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _resolve_model(self, model_id: str) -> tuple[ModelConfig, str]:
        """Validate model exists and is ready, return (config, local_path)."""
        config = self.manager.get_allowed().get(model_id)
        if not config:
            raise AppError(f"Model '{model_id}' is not in the allowed model list.", 404)

        reg = self.manager.get_registry().get(model_id)
        if not reg:
            raise AppError(f"Model '{model_id}' not downloaded. Pull it first.", 404)

        if reg.status == ModelStatus.DOWNLOADING:
            raise AppError(f"Model '{model_id}' is still downloading.", 409)

        if reg.status == ModelStatus.ERROR:
            raise AppError(f"Model '{model_id}' failed to download: {reg.error_message}", 500)

        if not reg.local_path or not Path(reg.local_path).exists():
            raise AppError(f"Model '{model_id}' files missing from disk. Re-pull required.", 404)

        return config, reg.local_path

    def load(self, model_id: str) -> None:
        """Load model weights into VRAM."""
        self.last_interaction = time.time()
        if self.active_id == model_id:
            return

        config, local_path = self._resolve_model(model_id)

        with self._lock:
            # Double-check after acquiring lock
            if self.active_id == model_id:
                return
            self.unload()

            device = "cuda" if _CUDA_AVAILABLE else "cpu"
            dtype = torch.bfloat16 if _CUDA_AVAILABLE and torch.cuda.is_bf16_supported() else torch.float16
            is_awq = "AWQ" in model_id.upper()
            attn_impl = "eager" if is_awq else "sdpa"

            logger.info("Loading %s (task=%s, dtype=%s, attn=%s)", model_id, config.task, dtype, attn_impl)

            try:
                if config.task == "text-generation":
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        local_path, device_map="auto", dtype=dtype, attn_implementation=attn_impl
                    )

                elif config.task == "image-to-text":
                    VLModel = _load_qwen_vl_model_class()
                    self.processor = AutoProcessor.from_pretrained(local_path)
                    self.model = VLModel.from_pretrained(
                        local_path, device_map="auto", dtype=dtype, attn_implementation=attn_impl
                    )

                elif config.task == "automatic-speech-recognition":
                    self.model = pipeline("automatic-speech-recognition", model=local_path, device=device, dtype=dtype)

                elif config.task in ("feature-extraction", "text-classification"):
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                    cls = AutoModel if config.task == "feature-extraction" else AutoModelForSequenceClassification
                    self.model = cls.from_pretrained(local_path, dtype=dtype).to(device)

                else:
                    raise AppError(f"Unsupported task '{config.task}' for model '{model_id}'.", 400)

            except AppError:
                raise
            except Exception as exc:
                self.model = None
                self.tokenizer = None
                self.processor = None
                if _CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                raise AppError(f"Failed to load model '{model_id}': {exc}", 500)

            self.active_id = model_id

    def chat(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        options: Dict[str, Any],
        stream: bool,
        is_chat: bool,
        force_json: bool,
    ) -> Union[Iterator[str], Dict[str, Any]]:
        """Text and multimodal generation with streaming support."""
        start_t = time.perf_counter()
        self.load(model_id)
        load_t = time.perf_counter() - start_t
        config = self.manager.get_allowed()[model_id]

        # --- Prepare inputs (isolated from generation) ---
        try:
            inputs, tok = self._prepare_inputs(config, messages, model_id)
        except AppError:
            raise
        except Exception as exc:
            logger.error("Inference preparation failed: %s", exc)
            if _CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            raise AppError(f"Input preparation failed: {exc}", 500)

        prompt_eval_count = inputs.input_ids.shape[1]
        eval_start_t = time.perf_counter()

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=STREAMER_TIMEOUT)
        gen_kwargs = self._build_gen_kwargs(inputs, options, streamer)

        # --- Generation thread with crash isolation ---
        gen_error: Optional[str] = None

        def _generate() -> None:
            nonlocal gen_error
            try:
                if self.model is None:
                    raise AppError("Model was unloaded before generation started.", 500)
                self.model.generate(**gen_kwargs)
            except Exception as exc:
                gen_error = f"{type(exc).__name__}: {exc}"
                logger.error("Generation thread crashed: %s", exc, exc_info=True)
                if _CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                try:
                    streamer.text_queue.put(streamer.stop_signal, timeout=2.0)
                except Exception:
                    pass

        threading.Thread(target=_generate, daemon=True).start()

        # --- Collect output ---
        def _collect() -> Iterator[Any]:
            with self.request_context():
                full_txt = ""
                thinking = ""
                in_thought = False
                t_count = 0
                first_tok_t: Optional[float] = None

                try:
                    for new_text in streamer:
                        if first_tok_t is None:
                            first_tok_t = time.perf_counter()
                        t_count += 1

                        # Parse <think> tags
                        if "<think>" in new_text:
                            in_thought = True
                            new_text = new_text.replace("<think>", "")
                        if "</think>" in new_text:
                            in_thought = False
                            new_text = new_text.replace("</think>", "")

                        if in_thought:
                            thinking += new_text
                        else:
                            full_txt += new_text

                        if stream:
                            chunk: Dict[str, Any] = {
                                "model": model_id,
                                "created_at": _utcnow_iso(),
                                "done": False,
                            }
                            if is_chat:
                                chunk["message"] = {"role": "assistant", "content": new_text}
                            else:
                                chunk["response"] = new_text
                            if thinking:
                                chunk["thinking"] = thinking
                            yield json.dumps(chunk) + "\n"

                except queue.Empty:
                    logger.error("Streamer timed out — model may have crashed.")
                    if _CUDA_AVAILABLE:
                        torch.cuda.empty_cache()

            # Final metrics (outside request_context so idle timer updates)
            eval_end = time.perf_counter()

            if force_json and not stream:
                full_txt = _extract_json(full_txt)

            res: Dict[str, Any] = {
                "model": model_id,
                "created_at": _utcnow_iso(),
                "done": True,
                "done_reason": "stop",
                "total_duration": int((eval_end - start_t) * 1e9),
                "load_duration": int(load_t * 1e9),
                "prompt_eval_count": prompt_eval_count,
                "eval_count": t_count,
                "eval_duration": int((eval_end - (first_tok_t or eval_start_t)) * 1e9),
            }

            if is_chat:
                res["message"] = {"role": "assistant", "content": full_txt}
            else:
                res["response"] = full_txt

            if thinking:
                res["thinking"] = thinking

            if gen_error:
                res["done_reason"] = "error"

            if stream:
                yield json.dumps(res) + "\n"
            else:
                yield res

        if stream:
            return _collect()
        return next(_collect())

    def _prepare_inputs(self, config: ModelConfig, messages: List[Dict[str, Any]], model_id: str) -> tuple[Any, Any]:
        """Tokenize / process inputs based on task type.

        FIXES APPLIED (compared to original):
        1. Detect actual MIME type instead of hardcoding image/jpeg
        2. Validate and normalize every image (EXIF, RGBA, corruption)
        3. Guard against process_vision_info returning None
        4. Per-image error isolation with indexed error messages
        5. VRAM cleanup on processor failure
        """
        if config.task == "image-to-text" and self.processor and self.model:
            process_vision_info = _load_qwen_vl_utils()
            max_px = self._dynamic_max_pixels()

            hf_msgs = []
            for m in messages:
                content: List[Dict[str, Any]] = [{"type": "text", "text": m["content"]}]
                for idx, img_b64 in enumerate(m.get("images") or []):
                    try:
                        # Validate, detect format, fix rotation, convert to RGB PNG
                        clean_bytes, mime_type = _safe_decode_base64_image(img_b64, idx)
                        clean_b64 = base64.b64encode(clean_bytes).decode("ascii")
                        content.append(
                            {
                                "type": "image",
                                "image": f"data:{mime_type};base64,{clean_b64}",
                                "max_pixels": max_px,
                            }
                        )
                    except AppError:
                        raise
                    except Exception as exc:
                        raise AppError(
                            f"Failed to process image [{idx}]: {type(exc).__name__}: {exc}",
                            400,
                            "image_processing_error",
                        )
                hf_msgs.append({"role": m["role"], "content": content})

            try:
                text = self.processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                raise AppError(f"Chat template application failed: {exc}", 500, "template_error")

            try:
                img_in, vid_in = process_vision_info(hf_msgs)
            except Exception as exc:
                logger.error("process_vision_info failed: %s", exc, exc_info=True)
                raise AppError(
                    f"Vision processing failed — image may be corrupt or unsupported: {exc}",
                    400,
                    "vision_processing_error",
                )

            # Guard against None returns
            if img_in is None:
                img_in = []
            if vid_in is None:
                vid_in = []

            try:
                inputs = self.processor(
                    text=[text],
                    images=img_in if img_in else None,
                    videos=vid_in if vid_in else None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            except Exception as exc:
                logger.error("Processor encoding failed: %s", exc, exc_info=True)
                if _CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                raise AppError(
                    f"Image encoding failed — likely corrupt or incompatible image: {exc}",
                    500,
                    "encoding_error",
                )

            return inputs, self.processor.tokenizer

        if self.tokenizer and self.model:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            return inputs, self.tokenizer

        raise AppError(f"Model '{model_id}' not properly loaded for text generation.", 500)

    def _build_gen_kwargs(self, inputs: Any, options: Dict[str, Any], streamer: Any) -> Dict[str, Any]:
        """Build generation kwargs from options."""
        processors = LogitsProcessorList([NaNSanitizerProcessor()])
        temp = options.get("temperature", 0.8)

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": options.get("num_predict", MAX_NEW_TOKENS_DEFAULT),
            "logits_processor": processors,
        }

        if temp <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            if "top_p" in options:
                gen_kwargs["top_p"] = options["top_p"]
            if "top_k" in options:
                gen_kwargs["top_k"] = options["top_k"]

        if options.get("seed") is not None:
            torch.manual_seed(options["seed"])

        return gen_kwargs

    @staticmethod
    def _dynamic_max_pixels() -> int:
        """Heuristic: cap image resolution based on free VRAM."""
        if not _CUDA_AVAILABLE:
            return 313600
        free_vram, _ = torch.cuda.mem_get_info()
        if free_vram < 4 * 1024**3:
            return 313600
        if free_vram < 8 * 1024**3:
            return 602112
        return 1003520

    def embeddings(
        self, model_id: str, texts: List[str], truncate: bool = True, dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate vector embeddings."""
        start = time.perf_counter()
        self.load(model_id)
        load_t = time.perf_counter() - start

        if not self.tokenizer or not self.model:
            raise AppError("Model not properly loaded for embeddings.", 500)

        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=truncate).to(self.model.device)
        except Exception as exc:
            raise AppError(f"Tokenization failed: {exc}", 400)

        if not truncate and hasattr(self.tokenizer, "model_max_length"):
            if inputs.input_ids.shape[1] > self.tokenizer.model_max_length:
                raise AppError(
                    f"Input exceeds context length ({self.tokenizer.model_max_length}). Set truncate=True.",
                    400,
                )

        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state.mean(dim=1)
            if dimensions is not None:
                output = output[:, :dimensions]
            embeddings_list = output.cpu().tolist()

        return {
            "model": model_id,
            "embeddings": embeddings_list,
            "total_duration": int((time.perf_counter() - start) * 1e9),
            "load_duration": int(load_t * 1e9),
            "prompt_eval_count": inputs.input_ids.numel(),
        }

    def rerank(self, model_id: str, query: str, docs: List[str]) -> Dict[str, Any]:
        """Calculate relevance scores for reranking."""
        start = time.perf_counter()
        self.load(model_id)
        if not self.tokenizer or not self.model:
            raise AppError("Model not properly loaded for reranking.", 500)

        pairs = [[query, doc] for doc in docs]
        inputs = self.tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.model.device
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1).float()
            scores = torch.sigmoid(logits).cpu().tolist()

        return {
            "model": model_id,
            "scores": scores,
            "total_duration": int((time.perf_counter() - start) * 1e9),
        }

    def transcribe(self, model_id: str, audio_path: str) -> Dict[str, Any]:
        """Process audio via ASR pipeline."""
        start = time.perf_counter()
        self.load(model_id)
        if not self.model:
            raise AppError("Model not properly loaded for transcription.", 500)

        try:
            res = self.model(audio_path, chunk_length_s=30, generate_kwargs={"condition_on_prev_tokens": False})
        except Exception as exc:
            raise AppError(f"Transcription failed: {exc}", 500)

        return {
            "model": model_id,
            "text": res.get("text", "").strip(),
            "total_duration": int((time.perf_counter() - start) * 1e9),
        }


def _extract_json(text: str) -> str:
    """Best-effort extraction of JSON from model output."""
    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw JSON
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# ===================================================================
# Idle monitor
# ===================================================================
async def _monitor_idle(eng: InferenceEngine) -> None:
    """Background task to release VRAM after idle timeout."""
    while True:
        await asyncio.sleep(15)

        with eng._lock:
            if not eng.active_id or eng.active_requests > 0 or eng.keep_alive_timeout < 0:
                continue
            should_unload = (time.time() - eng.last_interaction) > eng.keep_alive_timeout

        if should_unload:
            async with GPU_LOCK:
                eng.unload()


async def _handle_keep_alive(keep_alive: Union[str, int, None], model: str, eng: InferenceEngine) -> bool:
    """Process keep_alive: returns True if model was unloaded (caller should return early)."""
    timeout = _parse_keep_alive(keep_alive)
    if timeout is None:
        return False

    if timeout == 0.0:
        if eng.active_id == model:
            async with GPU_LOCK:
                eng.unload()
        return True

    with eng._lock:
        eng.keep_alive_timeout = timeout
    return False


# ===================================================================
# App setup
# ===================================================================
manager = ModelManager()
engine = InferenceEngine(manager)


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: start idle monitor, cleanup on shutdown."""
    task = asyncio.create_task(_monitor_idle(engine))
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    if engine.active_id:
        engine.unload()


app = FastAPI(title="Cudara", version=VERSION, lifespan=lifespan)


# ===================================================================
# Exception handlers
# ===================================================================
@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Structured AppError → JSON."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic validation → 400 JSON."""
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = " → ".join(str(location) for location in first.get("loc", []))
        msg = first.get("msg", str(exc))
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = str(exc)
    return JSONResponse(status_code=400, content={"error": f"Invalid request: {detail}"})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """HTTP exceptions → JSON."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def generic_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch-all → 500 JSON."""
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {type(exc).__name__}"})


# ===================================================================
# Middleware
# ===================================================================
@app.middleware("http")
async def ensure_json_content_type(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """Default to application/json for mutation methods (Ollama CLI compat)."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        ct = request.headers.get("content-type", "")
        if "application/json" not in ct:
            headers = [(k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"content-type"]
            headers.append((b"content-type", b"application/json"))
            request.scope["headers"] = headers
    return await call_next(request)


# ===================================================================
# Routes — Models
# ===================================================================
@app.get("/api/tags", tags=["Models"], summary="List local models", response_model=ListResponse)
async def list_models() -> ListResponse:
    """List models available in the local registry."""
    reg = manager.get_registry()
    allowed = manager.get_allowed()
    models = []

    for m_id, item in reg.items():
        if item.status != ModelStatus.READY or not item.local_path:
            continue
        model_path = Path(item.local_path)
        if not model_path.exists():
            continue

        config = allowed.get(m_id)
        family = config.task if config else "unknown"
        try:
            mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()
        except OSError:
            mtime = _utcnow_iso()

        models.append(
            ModelSummary(
                name=m_id,
                model=m_id,
                modified_at=mtime,
                size=_dir_size(model_path),
                digest=_make_digest(m_id),
                details=ModelDetails(
                    format="safetensors",
                    family=family,
                    families=[family],
                    parameter_size="unknown",
                    quantization_level="AWQ" if "AWQ" in m_id.upper() else "None",
                ),
            )
        )

    return ListResponse(models=models)


@app.post("/api/show", tags=["Models"], summary="Show model info", response_model=ShowResponse)
async def show_model(req: ShowRequest) -> ShowResponse:
    """Show model details and capabilities."""
    config = manager.get_allowed().get(req.model)
    reg_item = manager.get_registry().get(req.model)

    if not config or not reg_item or not reg_item.local_path:
        raise AppError(f"Model '{req.model}' not found.", 404)

    model_path = Path(reg_item.local_path)
    if not model_path.exists():
        raise AppError(f"Model '{req.model}' files missing from disk.", 404)

    info: Dict[str, Any] = {}
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                c = json.load(f)
                info = {
                    f"{config.task}.attention.head_count": c.get("num_attention_heads"),
                    f"{config.task}.context_length": c.get("max_position_embeddings"),
                    "general.architecture": c.get("model_type"),
                    "general.parameter_count": c.get("num_parameters", "unknown"),
                }
        except (json.JSONDecodeError, OSError):
            pass

    caps = ["completion"]
    if config.task == "image-to-text":
        caps.append("vision")
    if config.task == "automatic-speech-recognition":
        caps.append("audio")

    try:
        mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()
    except OSError:
        mtime = _utcnow_iso()

    return ShowResponse(
        parameters="temperature 0.7",
        license="See Hugging Face repository terms.",
        modified_at=mtime,
        details=ModelDetails(
            format="safetensors",
            family=config.task,
            families=[config.task],
            parameter_size="unknown",
            quantization_level="AWQ" if "AWQ" in req.model.upper() else "None",
        ),
        capabilities=caps,
        model_info=info,
        template=None,
    )


@app.get("/api/ps", tags=["System"], summary="List running models", response_model=PsResponse)
async def list_running() -> PsResponse:
    """Show currently loaded models and VRAM usage."""
    if not engine.active_id:
        return PsResponse(models=[])

    model_id = engine.active_id
    reg = manager.get_registry().get(model_id)
    allowed = manager.get_allowed().get(model_id)

    size = 0
    if reg and reg.local_path:
        size = _dir_size(Path(reg.local_path))

    expires_at = datetime.datetime.fromtimestamp(
        engine.last_interaction, tz=datetime.timezone.utc
    ) + datetime.timedelta(seconds=engine.keep_alive_timeout if engine.keep_alive_timeout >= 0 else 0)

    vram = torch.cuda.memory_allocated() if _CUDA_AVAILABLE else 0
    family = allowed.task if allowed else "unknown"

    return PsResponse(
        models=[
            Ps(
                name=model_id,
                model=model_id,
                size=size,
                digest=_make_digest(model_id),
                details=ModelDetails(
                    format="safetensors",
                    family=family,
                    families=[family],
                    parameter_size="unknown",
                    quantization_level="AWQ" if "AWQ" in model_id.upper() else "None",
                ),
                expires_at=expires_at.isoformat(),
                size_vram=vram,
                context_length=8192,
            )
        ]
    )


# ===================================================================
# Routes — Generation
# ===================================================================
@app.post("/api/generate", tags=["Generation"], summary="Generate a response", response_model=None)
async def generate(req: GenerateRequest, bg: BackgroundTasks) -> Union[JSONResponse, StreamingResponse]:
    """Generate text completion."""
    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(
            content={
                "model": req.model,
                "created_at": _utcnow_iso(),
                "response": "",
                "done": True,
                "done_reason": "unload",
            }
        )

    options_dict = _extract_options(req.options, req.logprobs, req.top_logprobs)

    async with GPU_LOCK:
        # --- Audio transcription path ---
        if options_dict.get("is_audio") and req.images:
            return await _handle_audio(req, options_dict)

        # --- Text generation path ---
        messages: List[Dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})

        prompt = req.prompt or ""
        if req.suffix:
            prompt = f"<|fim_prefix|>{prompt}<|fim_suffix|>{req.suffix}<|fim_middle|>"

        if prompt:
            msg: Dict[str, Any] = {"role": "user", "content": prompt}
            if req.images:
                msg["images"] = req.images
            messages.append(msg)

        if not messages:
            raise AppError("No prompt or system message provided.", 400)

        force_json = inject_json_instruction(messages, req.format)

        gen = await asyncio.to_thread(
            engine.chat,
            model_id=req.model,
            messages=messages,
            options=options_dict,
            stream=req.stream,
            is_chat=False,
            force_json=force_json,
        )

        if req.stream:
            return StreamingResponse(gen, media_type="application/x-ndjson")  # type: ignore
        return JSONResponse(content=gen)  # type: ignore


async def _handle_audio(req: GenerateRequest, options_dict: Dict[str, Any]) -> JSONResponse:
    """Handle audio transcription via the generate endpoint."""
    if not req.images:
        raise AppError("No audio data provided in images field.", 400)

    try:
        audio_bytes = base64.b64decode(req.images[0])
    except Exception:
        raise AppError("Invalid base64 audio data.", 400)

    # Detect format from magic bytes
    ext = "wav"
    if audio_bytes.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3")):
        ext = "mp3"
    elif audio_bytes.startswith(b"OggS"):
        ext = "ogg"
    elif len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp":
        ext = "m4a"

    tmp = TEMP_DIR / f"asr_{int(time.time())}_{id(audio_bytes) % 10000}.{ext}"
    try:
        with open(tmp, "wb") as f:
            f.write(audio_bytes)
        res = await asyncio.to_thread(engine.transcribe, req.model, str(tmp))
        return JSONResponse(content={"model": req.model, "response": res["text"], "done": True})
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/chat", tags=["Generation"], summary="Generate a chat message", response_model=None)
async def chat(req: ChatRequest) -> Union[JSONResponse, StreamingResponse]:
    """Generate the next message in a conversation."""
    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(
            content={
                "model": req.model,
                "created_at": _utcnow_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "unload",
            }
        )

    options_dict = _extract_options(req.options, req.logprobs, req.top_logprobs)

    async with GPU_LOCK:
        msgs = [m.model_dump(exclude_unset=True) for m in req.messages]
        force_json = inject_json_instruction(msgs, req.format)

        gen = await asyncio.to_thread(
            engine.chat,
            model_id=req.model,
            messages=msgs,
            options=options_dict,
            stream=req.stream,
            is_chat=True,
            force_json=force_json,
        )

        if req.stream:
            return StreamingResponse(gen, media_type="application/x-ndjson")  # type: ignore
        return JSONResponse(content=gen)  # type: ignore


@app.post("/api/embed", tags=["Generation"], summary="Generate embeddings", response_model=EmbedResponse)
@app.post("/api/embeddings", include_in_schema=False)
async def embed(req: EmbedRequest) -> JSONResponse:
    """Generate embeddings or rerank scores."""
    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(content={"model": req.model, "done": True})

    options_dict = _extract_options(req.options)

    async with GPU_LOCK:
        if options_dict.get("is_rerank") and isinstance(req.input, list) and len(req.input) > 1:
            res = await asyncio.to_thread(engine.rerank, req.model, req.input[0], req.input[1:])
            return JSONResponse(content=res)

        texts = [req.input] if isinstance(req.input, str) else req.input
        res = await asyncio.to_thread(
            engine.embeddings,
            model_id=req.model,
            texts=texts,
            truncate=req.truncate,
            dimensions=req.dimensions,
        )
        return JSONResponse(content=res)


# ===================================================================
# Routes — Pull / Delete / System
# ===================================================================
@app.post("/api/pull", tags=["Models"], summary="Pull a model", response_model=None)
async def pull_model(req: PullRequest, bg: BackgroundTasks) -> Union[Dict[str, str], StreamingResponse]:
    """Download model weights from Hugging Face."""
    if req.model not in manager.get_allowed():
        raise AppError(f"Model '{req.model}' is not in the allowed list (models.json).", 400)

    if not req.stream:
        bg.add_task(manager.download_task, req.model)
        return {"status": "success"}

    async def _progress() -> AsyncIterator[str]:
        threading.Thread(target=manager.download_task, args=(req.model,), daemon=True).start()
        last_status = None
        for _ in range(3600):  # Max 1 hour polling
            reg = manager.get_registry().get(req.model)
            if not reg:
                yield json.dumps({"status": "initializing..."}) + "\n"
            else:
                event = {"status": reg.status.value, "digest": _make_digest(req.model)}
                if reg.status == ModelStatus.READY:
                    event["status"] = "success"
                    yield json.dumps(event) + "\n"
                    return
                if reg.status == ModelStatus.ERROR:
                    event["status"] = f"error: {reg.error_message}"
                    yield json.dumps(event) + "\n"
                    return
                if reg.status.value != last_status:
                    yield json.dumps(event) + "\n"
                    last_status = reg.status.value
            await asyncio.sleep(1)

    return StreamingResponse(_progress(), media_type="application/x-ndjson")


@app.delete("/api/delete", tags=["Models"], summary="Delete a model")
async def delete_model(req: DeleteRequest) -> JSONResponse:
    """Remove model files and registry entry."""
    if engine.active_id == req.model:
        async with GPU_LOCK:
            engine.unload()
    try:
        manager.delete_model(req.model)
        return JSONResponse(content={"status": "success"})
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"Failed to delete '{req.model}': {exc}", 500)


@app.get("/api/version", tags=["System"], summary="Server version")
def get_version() -> Dict[str, str]:
    """Return server version."""
    return {"version": VERSION}


@app.get("/health", tags=["System"], summary="Health check")
async def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ===================================================================
# Stub endpoints (Ollama compat)
# ===================================================================
@app.post("/api/create", tags=["Models"], include_in_schema=False)
async def create_stub() -> JSONResponse:
    """Stub: create model from Modelfile."""
    return JSONResponse(status_code=501, content={"error": "Not implemented."})


@app.post("/api/copy", tags=["Models"], include_in_schema=False)
async def copy_stub() -> JSONResponse:
    """Stub: alias a model."""
    return JSONResponse(status_code=501, content={"error": "Not implemented."})


@app.post("/api/push", tags=["Models"], include_in_schema=False)
async def push_stub() -> JSONResponse:
    """Stub: push model to registry."""
    return JSONResponse(status_code=501, content={"error": "Not implemented."})
