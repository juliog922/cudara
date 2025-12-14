"""
Cudara - CUDA Inference Server
==============================
Ollama-compatible API for HuggingFace models.
"""

import base64
import gc
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import hf_hub_download, login, snapshot_download
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    AutoModel,
    BitsAndBytesConfig,
    pipeline,
)

# Internal imports
from src.cudara.quantization import (
    ModelQuantizer,
    ensure_video_preprocessor_json,
    load_processor_with_fallback,
    sync_token_ids_with_tokenizer,
)

# GGUF Support
try:
    from llama_cpp import Llama

    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*num_logits_to_keep.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")


@contextmanager
def suppress_output(suppress: bool = True):
    """
    Context manager to suppress stdout/stderr for noisy libraries.

    Parameters
    ----------
    suppress : bool
        If False, does nothing.
    """
    if not suppress:
        yield
        return

    class Filter:
        def __init__(self, stream, patterns):
            self.stream = stream
            self.patterns = patterns
            self._skip = False

        def write(self, text):
            if any(p in text.lower() for p in self.patterns):
                self._skip = True
                return
            if self._skip and text in ("\n", "\r\n", "\r"):
                self._skip = False
                return
            self._skip = False
            self.stream.write(text)

        def flush(self):
            self.stream.flush()

    patterns = [
        "flash attention",
        "gptq",
        "exllama",
        "bitsandbytes",
        "compiling",
        "loading checkpoint",
    ]
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Filter(old_stdout, patterns)
    sys.stderr = Filter(old_stderr, patterns)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# =============================================================================
# LOGGING
# =============================================================================
class ConciseFormatter(logging.Formatter):
    """Custom log formatter for cleaner console output."""

    FORMATS = {
        logging.DEBUG: "\033[90m%(message)s\033[0m",
        logging.INFO: "%(message)s",
        logging.WARNING: "\033[33m⚠ %(message)s\033[0m",
        logging.ERROR: "\033[31m✗ %(message)s\033[0m",
        logging.CRITICAL: "\033[31;1m✗ %(message)s\033[0m",
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, "%(message)s")
        return logging.Formatter(fmt).format(record)


def setup_logging():
    """Configure root logger with concise formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(ConciseFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for lib in ["transformers", "huggingface_hub", "urllib3", "httpx"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("cudara")


logger = setup_logging()

# =============================================================================
# CONFIG
# =============================================================================
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
MODELS_DIR: Path = Path("models")
REGISTRY_FILE: Path = Path("registry.json")
ALLOWED_MODELS_FILE: Path = Path("models.json")
TEMP_DIR: Path = Path("temp_uploads")

MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

QWEN_VL_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% set image_count.value = image_count.value + 1 %}"
    "{% elif content['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% set video_count.value = video_count.value + 1 %}"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# =============================================================================
# ERROR HANDLING
# =============================================================================
class AppError(Exception):
    """
    Standard application exception.

    Attributes
    ----------
    message : str
        Error description.
    status_code : int
        HTTP status code.
    details : dict
        Additional error context (code, etc).
    """

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ErrorCode:
    """Standard error codes for API responses."""

    MODEL_NOT_FOUND = "model_not_found"
    MODEL_NOT_READY = "model_not_ready"
    MODEL_NOT_ALLOWED = "model_not_allowed"
    INFERENCE_ERROR = "inference_error"
    INVALID_REQUEST = "invalid_request"
    BACKEND_ERROR = "backend_error"


def error_response(code: str, message: str, **kwargs) -> dict:
    """Format standard error response dict."""
    return {"error": {"code": code, "message": message, **kwargs}}


# =============================================================================
# DATA MODELS
# =============================================================================
class ModelStatus(str, Enum):
    """Model lifecycle states."""

    DOWNLOADING = "downloading"
    QUANTIZING = "quantizing"
    READY = "ready"
    ERROR = "error"


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""

    enabled: bool = False
    prequantize: bool = False
    method: str = "bitsandbytes"
    bits: int = 4
    category: str = "text_llm_medium"
    skip_modules: Optional[List[str]] = None


class ImageProcessingCfg(BaseModel):
    """Configuration for image preprocessing."""

    min_pixels: int = 200704
    optimal_pixels: int = 602112
    max_pixels: int = 802816
    max_retry_attempts: int = 3


class InferenceOptimization(BaseModel):
    """Runtime optimization flags."""

    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    use_static_cache: bool = True


class ModelConfig(BaseModel):
    """
    Configuration definition for an allowed model.

    Attributes
    ----------
    task : str
        HF task identifier (text-generation, etc).
    backend : str
        'transformers' or 'gguf'.
    architecture : str
        Model class to load.
    quantization : QuantizationConfig
        Quantization settings.
    """

    description: Optional[str] = None
    task: str = "feature-extraction"
    backend: str = "transformers"
    filename: Optional[str] = None
    architecture: str = "AutoModel"
    dtype: str = "auto"
    trust_remote_code: bool = False
    quantization: Optional[QuantizationConfig] = None
    parameters: Optional[Dict[str, Any]] = {}
    generation_defaults: Optional[Dict[str, Any]] = {}
    default_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    stop_strings: Optional[List[str]] = None
    inference_optimization: Optional[InferenceOptimization] = None
    image_processing: Optional[ImageProcessingCfg] = None


class RegistryItem(BaseModel):
    """Runtime state of a downloaded model."""

    status: ModelStatus
    local_path: Optional[str] = None
    error_message: Optional[str] = None
    is_prequantized: Optional[bool] = False
    backend: Optional[str] = "transformers"


class GenerateRequest(BaseModel):
    """Request schema for /api/generate."""

    model: str
    prompt: str
    system: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Single message in chat history."""

    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Request schema for /api/chat."""

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request schema for /api/embeddings."""

    model: str
    input: Union[str, List[str]]
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PullRequest(BaseModel):
    """Request schema for /api/pull."""

    name: str
    stream: bool = False


class ModelIdentifier(BaseModel):
    """Simple model for endpoints requiring just a name."""

    name: str = Field(..., description="The model identifier (e.g., 'Qwen/Qwen2.5-3B')")


# =============================================================================
# MODEL MANAGER
# =============================================================================
class ModelManager:
    """
    Manages model configuration, downloading, and registry state.

    Attributes
    ----------
    quantizer : ModelQuantizer
        Helper for handling quantization during download.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.quantizer = ModelQuantizer(MODELS_DIR)
        self._init_files()

    def _init_files(self):
        """Ensure necessary JSON config files exist."""
        if not REGISTRY_FILE.exists():
            self._save_json(REGISTRY_FILE, {})
        if not ALLOWED_MODELS_FILE.exists():
            self._save_json(ALLOWED_MODELS_FILE, {})

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Thread-safe JSON load."""
        with self._lock:
            if not path.exists():
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def _save_json(self, path: Path, data: Dict[str, Any]):
        """Thread-safe JSON save."""
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def get_allowed_models(self) -> Dict[str, ModelConfig]:
        """
        Get list of models configured in models.json.

        Returns
        -------
        Dict[str, ModelConfig]
            Map of model_id to configuration.
        """
        data = self._load_json(ALLOWED_MODELS_FILE)
        results = {}
        for k, v in data.items():
            if k.startswith("_"):
                continue
            if "quantization" in v:
                v["quantization"] = QuantizationConfig(**v["quantization"])
            if "inference_optimization" in v:
                v["inference_optimization"] = InferenceOptimization(**v["inference_optimization"])
            if "image_processing" in v:
                v["image_processing"] = ImageProcessingCfg(**v["image_processing"])
            results[k] = ModelConfig(**v)
        return results

    def get_registry(self) -> Dict[str, RegistryItem]:
        """
        Get runtime status of all models.

        Returns
        -------
        Dict[str, RegistryItem]
            Map of model_id to download/status state.
        """
        data = self._load_json(REGISTRY_FILE)
        return {k: RegistryItem(**v) for k, v in data.items()}

    def update_registry(
        self,
        model_id: str,
        status: ModelStatus,
        path: str = None,
        error: str = None,
        is_prequantized: bool = False,
        backend: str = "transformers",
    ):
        """Update the registry entry for a model."""
        raw = self._load_json(REGISTRY_FILE)
        entry = raw.get(model_id, {})
        entry.update(
            {"status": status.value, "is_prequantized": is_prequantized, "backend": backend}
        )
        if path:
            entry["local_path"] = path
        if error:
            entry["error_message"] = error
        raw[model_id] = entry
        self._save_json(REGISTRY_FILE, raw)

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get local filesystem path if model is READY."""
        reg = self.get_registry()
        if model_id in reg and reg[model_id].status == ModelStatus.READY:
            return Path(reg[model_id].local_path)
        return None

    def is_prequantized(self, model_id: str) -> bool:
        """Check if model was pre-quantized during download."""
        reg = self.get_registry()
        return reg[model_id].is_prequantized if model_id in reg else False

    def download_model_task(self, model_id: str):
        """
        Background task to download and optionally quantize a model.

        Parameters
        ----------
        model_id : str
            ID of the model to download.
        """
        config = self.get_allowed_models().get(model_id)
        if not config:
            return

        target_dir = MODELS_DIR / model_id.replace("/", "--").replace(".", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = target_dir.parent / f"{target_dir.name}_temp"

        try:
            logger.info(f"Downloading {model_id}...")

            if config.backend == "gguf":
                if not config.filename:
                    raise ValueError("Filename required for GGUF")
                local_path = hf_hub_download(
                    repo_id=model_id, filename=config.filename, local_dir=target_dir, token=HF_TOKEN
                )
                self.update_registry(model_id, ModelStatus.READY, str(local_path), backend="gguf")
            else:
                should_quantize = (
                    config.quantization
                    and config.quantization.enabled
                    and config.quantization.prequantize
                )
                dest = temp_dir if should_quantize else target_dir
                snapshot_download(repo_id=model_id, local_dir=dest, token=HF_TOKEN)

                if should_quantize:
                    logger.info(f"Quantizing {model_id}...")
                    self.update_registry(model_id, ModelStatus.QUANTIZING)
                    res = self.quantizer.prequantize_model(
                        model_id,
                        temp_dir,
                        target_dir,
                        task=config.task,
                        trust_remote_code=config.trust_remote_code,
                        architecture=config.architecture,
                    )
                    self.update_registry(
                        model_id,
                        ModelStatus.READY,
                        str(target_dir),
                        is_prequantized=res.get("quantized", False),
                    )
                else:
                    self.update_registry(model_id, ModelStatus.READY, str(target_dir))

            logger.info(f"✓ {model_id} ready")

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            self.update_registry(model_id, ModelStatus.ERROR, error=str(e))
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def delete_model(self, model_id: str):
        """Delete model files and registry entry."""
        path = self.get_model_path(model_id)
        if path and path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

        reg = self._load_json(REGISTRY_FILE)
        if model_id in reg:
            del reg[model_id]
            self._save_json(REGISTRY_FILE, reg)

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Construct detailed info for `show` command."""
        config = self.get_allowed_models().get(model_id)
        registry = self.get_registry().get(model_id)
        if not config:
            return None
        return {
            "modelfile": f"# {model_id}\n# {config.description or ''}",
            "parameters": json.dumps(config.generation_defaults or {}, indent=2),
            "template": config.system_prompt or "",
            "details": {
                "format": config.backend,
                "family": config.architecture,
                "quantization_level": f"{config.quantization.bits}bit"
                if config.quantization
                else "none",
            },
            "model_info": {
                "task": config.task,
                "status": registry.status.value if registry else "not_downloaded",
            },
        }


# =============================================================================
# INFERENCE ENGINE
# =============================================================================
class InferenceEngine:
    """
    Handles model loading, unloading, and inference execution.

    Features:
    - Lazy loading
    - Automatic unloading on idle (300s)
    - Supports GGUF and Transformers
    - Vision support
    - **Concurrency Safety**: Strict locking for GPU operations.
    """

    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.active_pipeline: Any = None
        self.active_model_id = None
        self.active_config: Optional[ModelConfig] = None
        self.active_dtype: Optional[torch.dtype] = None
        self.last_access = 0.0

        # _lock protects internal state changes (loading/unloading)
        self._lock = threading.RLock()
        # _inference_lock forces strictly sequential execution of inference
        # This prevents model swapping during a generation request and ensures single-user GPU access.
        self._inference_lock = threading.Lock()

        threading.Thread(target=self._idle_monitor, daemon=True).start()

    def _idle_monitor(self):
        """Background thread to unload models after inactivity."""
        while True:
            time.sleep(60)
            # We use _lock here to check state, but we don't need _inference_lock
            # just to check time. We only lock if we decide to unload.
            with self._lock:
                if self.active_pipeline and (time.time() - self.last_access > 300):
                    # Ensure no inference is running before unloading
                    if self._inference_lock.acquire(blocking=False):
                        try:
                            self._unload()
                        finally:
                            self._inference_lock.release()
                    else:
                        logger.info("Skipping idle unload: inference in progress")

    def _unload(self):
        """Unload current model and free VRAM."""
        self.active_pipeline = None
        self.active_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded (idle)")

    def _clean_reasoning(self, text: str) -> str:
        """Strip reasoning/thought chains from output if present."""
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def load_model(self, model_id: str):
        """
        Load a model into memory if not already active.
        This method is protected by self._lock via the caller (inference methods).

        Parameters
        ----------
        model_id : str
            Model to load.
        """
        # Note: We don't acquire _lock here explicitly because this function is
        # called inside the _inference_lock block of the public methods.
        # The _inference_lock implies exclusive access to the GPU.

        if self.active_model_id == model_id:
            return

        config = self.manager.get_allowed_models().get(model_id)
        if not config:
            raise AppError(
                f"Model '{model_id}' not configured", 404, {"code": ErrorCode.MODEL_NOT_ALLOWED}
            )

        path = self.manager.get_model_path(model_id)
        if not path:
            raise AppError(
                f"Model '{model_id}' not downloaded", 404, {"code": ErrorCode.MODEL_NOT_READY}
            )

        if self.active_pipeline:
            self._unload()

        # GGUF Backend
        if config.backend == "gguf":
            if not GGUF_AVAILABLE:
                raise AppError(
                    "llama-cpp-python not installed", 500, {"code": ErrorCode.BACKEND_ERROR}
                )

            logger.info(f"Loading {model_id} (GGUF)...")
            params = config.parameters or {}
            is_vl = "vl" in model_id.lower() and "qwen" in model_id.lower()

            try:
                model = Llama(
                    model_path=str(path),
                    n_gpu_layers=params.get("n_gpu_layers", -1),
                    n_ctx=params.get("n_ctx", 12288 if is_vl else 4096),
                    n_batch=params.get("n_batch", 512),
                    verbose=False,
                    chat_format="qwen2-vl" if is_vl else None,
                )
                self.active_pipeline = model
                self.active_model_id = model_id
                self.active_config = config
                logger.info(f"✓ {model_id} loaded")
                return
            except Exception as e:
                raise AppError(f"GGUF load failed: {e}", 500, {"code": ErrorCode.BACKEND_ERROR})

        # Transformers Backend
        ensure_video_preprocessor_json(path)
        is_prequantized = self.manager.is_prequantized(model_id)

        if config.dtype == "float32":
            dtype = torch.float32
        elif config.dtype == "float16":
            dtype = torch.float16
        elif config.dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.active_dtype = dtype
        logger.info(f"Loading {model_id}...")

        model_cls = getattr(transformers, config.architecture, AutoModel)
        load_kwargs = {
            "device_map": "auto",
            "dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if config.task != "automatic-speech-recognition":
            try:
                from transformers.utils import is_flash_attn_2_available

                if is_flash_attn_2_available():
                    load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass

        if not is_prequantized and config.quantization and config.quantization.enabled:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(config.quantization.bits == 4),
                load_in_8bit=(config.quantization.bits == 8),
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=config.quantization.skip_modules or ["lm_head"],
            )

        with suppress_output():
            model = model_cls.from_pretrained(str(path), **load_kwargs)
        model.eval()

        processor = load_processor_with_fallback(path, config.trust_remote_code)
        is_vlm = config.task == "image-to-text" or "vl" in model_id.lower()
        tokenizer_ref = getattr(processor, "tokenizer", processor)

        if is_vlm:
            tokenizer_ref.chat_template = QWEN_VL_TEMPLATE
            if hasattr(processor, "chat_template"):
                processor.chat_template = QWEN_VL_TEMPLATE
            sync_token_ids_with_tokenizer(model, processor)
        else:
            if not tokenizer_ref.pad_token:
                tokenizer_ref.pad_token = tokenizer_ref.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        if config.task == "automatic-speech-recognition":
            self.active_pipeline = pipeline(
                task=config.task,
                model=model,
                tokenizer=tokenizer_ref,
                feature_extractor=getattr(processor, "feature_extractor", processor),
                dtype=dtype,
                device_map="auto",
            )
        else:
            self.active_pipeline = (model, processor)

        self.active_model_id = model_id
        self.active_config = config
        self.last_access = time.time()
        logger.info(f"✓ {model_id} loaded")

    def generate(
        self,
        model_id: str,
        prompt: str,
        images: List[str] = None,
        system: str = None,
        options: dict = None,
    ) -> dict:
        """
        Execute text generation.
        Strictly serialized to prevent concurrent GPU usage.
        """
        with self._inference_lock:
            self.load_model(model_id)
            config = self.active_config
            start = time.perf_counter()
            self.last_access = time.time()
            options = options or {}

            # GGUF
            if config.backend == "gguf":
                return self._generate_gguf(prompt, images, system, options, start)

            # Transformers
            return self._generate_transformers(prompt, images, system, options, start)

    def _generate_gguf(self, prompt, images, system, options, start):
        config = self.active_config
        model = self.active_pipeline

        messages = []
        if system or config.system_prompt:
            messages.append({"role": "system", "content": system or config.system_prompt})

        user_content = []
        if images and config.task == "image-to-text":
            for img in images:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                )
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        if len(user_content) > 1 or (user_content and user_content[0].get("type") == "image_url"):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt or ""})

        gen_params = {**(config.generation_defaults or {}), **options}

        try:
            output = model.create_chat_completion(
                messages=messages,
                max_tokens=gen_params.get("num_predict", gen_params.get("max_new_tokens", 256)),
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                stop=config.stop_strings or [],
            )
            result = self._clean_reasoning(output["choices"][0]["message"]["content"])
            usage = output.get("usage", {})
            elapsed = time.perf_counter() - start

            return {
                "model": self.active_model_id,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "response": result,
                "done": True,
                "total_duration": int(elapsed * 1e9),
                "eval_count": usage.get("completion_tokens", 0),
            }
        except Exception as e:
            raise AppError(f"Generation failed: {e}", 500, {"code": ErrorCode.INFERENCE_ERROR})

    def _generate_transformers(self, prompt, images, system, options, start):
        config = self.active_config
        model, processor = self.active_pipeline
        tokenizer = getattr(processor, "tokenizer", processor)

        messages = []
        if system or config.system_prompt:
            messages.append({"role": "system", "content": system or config.system_prompt})

        content = []
        image_obj = None

        if images and config.task == "image-to-text":
            import io

            from src.cudara.image_processing import AdaptiveImageProcessor

            img_data = base64.b64decode(images[0])
            raw_img = Image.open(io.BytesIO(img_data)).convert("RGB")

            model_usage = (
                torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            )
            img_proc = AdaptiveImageProcessor.from_gpu_info(model_usage)
            if config.image_processing:
                img_proc.config.min_pixels = config.image_processing.min_pixels
                img_proc.config.optimal_pixels = config.image_processing.optimal_pixels

            image_obj, _ = img_proc.prepare_for_inference(raw_img)
            content.append({"type": "image", "image": image_obj})

        if prompt:
            content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content if image_obj else prompt})

        try:
            if config.parameters and config.parameters.get("use_qwen_vision_utils"):
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=[image_obj] if image_obj else None,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if image_obj:
                    inputs = processor(
                        text=[text], images=[image_obj], padding=True, return_tensors="pt"
                    )
                else:
                    inputs = tokenizer(text, return_tensors="pt")
        except Exception as e:
            logger.warning(f"Template failed: {e}")
            inputs = tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {**(config.generation_defaults or {}), **options}
        gen_kwargs["use_cache"] = True
        if "num_predict" in gen_kwargs:
            gen_kwargs["max_new_tokens"] = gen_kwargs.pop("num_predict")

        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        gen_time = time.perf_counter() - gen_start

        new_tokens = outputs[0][input_len:]
        result = self._clean_reasoning(tokenizer.decode(new_tokens, skip_special_tokens=True))

        if config.stop_strings:
            for s in config.stop_strings:
                result = result.split(s)[0]

        elapsed = time.perf_counter() - start

        return {
            "model": self.active_model_id,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "response": result.strip(),
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": len(new_tokens),
            "eval_duration": int(gen_time * 1e9),
        }

    def chat(self, model_id: str, messages: List[ChatMessage], options: dict = None) -> dict:
        """
        Execute chat interaction.
        Note: The actual locking happens in `self.generate` which this calls,
        but we add it here too if we ever change implementation to avoid ambiguity.
        Since `generate` locks, this is safe, but for clarity:
        """
        # We process messages here then call generate.
        # generate() will acquire the lock.
        last_msg = messages[-1]
        system = next((m.content for m in messages if m.role == "system"), None)
        return self.generate(model_id, last_msg.content, last_msg.images, system, options)

    def embeddings(self, model_id: str, texts: List[str], options: dict = None) -> dict:
        """
        Generate embeddings.
        Strictly serialized.
        """
        with self._inference_lock:
            self.load_model(model_id)
            config = self.active_config
            start = time.perf_counter()
            self.last_access = time.time()

            if config.task != "feature-extraction":
                raise AppError(
                    f"Model {model_id} is not an embedding model",
                    400,
                    {"code": ErrorCode.INVALID_REQUEST},
                )

            model, tokenizer = self.active_pipeline
            all_embeddings = []

            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).float().cpu().numpy().tolist()[0]
                all_embeddings.append(emb)

            return {
                "model": model_id,
                "embeddings": all_embeddings,
                "total_duration": int((time.perf_counter() - start) * 1e9),
            }

    def transcribe(self, model_id: str, audio_path: str, options: dict = None) -> dict:
        """
        Transcribe audio file.
        Strictly serialized.
        """
        with self._inference_lock:
            self.load_model(model_id)
            config = self.active_config
            start = time.perf_counter()
            self.last_access = time.time()

            if config.task != "automatic-speech-recognition":
                raise AppError(
                    f"Model {model_id} is not an ASR model", 400, {"code": ErrorCode.INVALID_REQUEST}
                )

            options = options or {}
            merged = {**(config.generation_defaults or {}), **options}
            pipeline_args = {
                k: v
                for k, v in merged.items()
                if k in ["return_timestamps", "chunk_length_s", "stride_length_s"]
            }
            generate_kwargs = {k: v for k, v in merged.items() if k not in pipeline_args}
            generate_kwargs.pop("return_token_timestamps", None)

            try:
                res = self.active_pipeline(audio_path, generate_kwargs=generate_kwargs, **pipeline_args)
                if isinstance(res, list):
                    text = " ".join([c.get("text", "") for c in res])
                elif isinstance(res, dict):
                    text = res.get("text", "")
                else:
                    text = str(res)

                return {
                    "model": model_id,
                    "text": text.strip(),
                    "total_duration": int((time.perf_counter() - start) * 1e9),
                }
            except Exception as e:
                raise AppError(f"Transcription failed: {e}", 500, {"code": ErrorCode.INFERENCE_ERROR})


# =============================================================================
# API
# =============================================================================
manager = ModelManager()
engine = InferenceEngine(manager)

app = FastAPI(
    title="Cudara",
    description="""
    **Lightweight CUDA inference server.** This API is compatible with the Ollama standard, allowing you to drop it in
    as a replacement backend for existing tools.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Cudara Maintainer",
        "url": "https://github.com/juliog922/cudara",
    },
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.details.get("code", "error"), exc.message),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}")
    return JSONResponse(status_code=500, content=error_response("internal_error", str(exc)))


@app.get("/", tags=["UI"], summary="Landing Page", response_class=HTMLResponse)
async def landing_page(request: Request):
    """
    **Serves the UI.**

    If the client accepts `text/html`, serves `index.html`.
    Otherwise, falls back to the JSON health check.
    """
    landing_file = Path("index.html")
    accept = (request.headers.get("accept") or "").lower()
    if "text/html" in accept and landing_file.exists():
        return HTMLResponse(content=landing_file.read_text(encoding="utf-8"))
    return JSONResponse(content=health())


@app.get(
    "/health",
    tags=["Health"],
    summary="Server Health",
    response_description="Server status and VRAM usage",
)
def health():
    """
    **Health Check.**

    Returns the running status, version, currently loaded model, and VRAM consumption.
    Useful for readiness probes in container orchestrators.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_model": engine.active_model_id,
        "cuda_available": torch.cuda.is_available(),
        "vram_used": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
        if torch.cuda.is_available()
        else "0GB",
    }


@app.get("/api/tags", tags=["Models"], summary="List Models")
def list_models():
    """
    **List Available Models.**

    Returns a list of models configured in `models.json` and their local download status.
    Compatible with Ollama's `/api/tags`.
    """
    allowed = manager.get_allowed_models()
    registry = manager.get_registry()
    models = []
    for model_id, config in allowed.items():
        reg = registry.get(model_id)
        models.append(
            {
                "name": model_id,
                "modified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "size": 0,  # TODO: Calculate actual disk size
                "details": {"format": config.backend, "family": config.architecture},
                "status": reg.status.value if reg else "not_downloaded",
                "description": config.description,
            }
        )
    return {"models": models}


@app.post("/api/show", tags=["Models"], summary="Model Details")
def show_model(request: ModelIdentifier):
    """
    **Show Model Information.**

    Returns details about a specific model, including its Modelfile parameters,
    template, and quantization status.
    """
    model_id = request.name
    info = manager.get_model_info(model_id)
    if not info:
        raise AppError(f"Model '{model_id}' not found", 404, {"code": ErrorCode.MODEL_NOT_FOUND})
    return info


@app.post("/api/pull", tags=["Models"], summary="Pull Model")
def pull_model(request: PullRequest, bg: BackgroundTasks):
    """
    **Download/Pull a Model.**

    Triggers a background task to download (and optionally quantize) the model
    from HuggingFace.

    - **name**: The model ID (must be in `models.json`).
    """
    if request.name not in manager.get_allowed_models():
        raise AppError(
            f"Model '{request.name}' not allowed", 403, {"code": ErrorCode.MODEL_NOT_ALLOWED}
        )
    manager.update_registry(request.name, ModelStatus.DOWNLOADING)
    bg.add_task(manager.download_model_task, request.name)
    return {"status": "downloading"}


@app.delete("/api/delete", tags=["Models"], summary="Delete Model")
def delete_model(request: ModelIdentifier):
    """
    **Delete a Model.**

    Removes the model files from disk and unloads it from VRAM if active.
    """
    model_id = request.name
    if engine.active_model_id == model_id:
        engine._unload()
    manager.delete_model(model_id)
    return {"status": "deleted"}


@app.post("/api/generate", tags=["Inference"], summary="Generate Text")
def generate(request: GenerateRequest):
    """
    **Text Generation.**

    Standard completion endpoint.

    - **model**: Model ID.
    - **prompt**: Input text.
    - **images**: List of base64 strings (for VLMs).
    - **options**: Inference parameters (temperature, top_p, etc).
    """
    return engine.generate(
        request.model, request.prompt, request.images, request.system, request.options
    )


@app.post("/api/chat", tags=["Inference"], summary="Chat Completion")
def chat(request: ChatRequest):
    """
    **Chat Interface.**

    Interactive chat completion compatible with OpenAI/Ollama formats.

    - **messages**: List of message objects `{"role": "user", "content": "..."}`.
    """
    result = engine.chat(request.model, request.messages, request.options)
    return {
        "model": result["model"],
        "created_at": result["created_at"],
        "message": {"role": "assistant", "content": result["response"]},
        "done": True,
        "total_duration": result["total_duration"],
        "eval_count": result.get("eval_count", 0),
    }


@app.post("/api/embeddings", tags=["Inference"], summary="Generate Embeddings")
@app.post("/api/embed", tags=["Inference"], include_in_schema=False)
def embeddings(request: EmbeddingRequest):
    """
    **Feature Extraction.**

    Generates vector embeddings for the input text(s).
    Requires a model with task `feature-extraction`.
    """
    texts = request.input if isinstance(request.input, list) else [request.input]
    return engine.embeddings(request.model, texts, request.options)


@app.post("/api/transcribe", tags=["Inference"], summary="Audio Transcription")
async def transcribe(
    model: str = Form(..., description="ASR Model ID"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a)"),
    options: str = Form("{}", description="JSON string of generation options"),
):
    """
    **Speech-to-Text.**

    Multipart upload for audio transcription.

    - **model**: Must be an `automatic-speech-recognition` model (e.g., Whisper).
    - **options**: Can include `chunk_length_s`, `return_timestamps`, etc.
    """
    temp_path = TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return engine.transcribe(model, str(temp_path), json.loads(options))
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/vision", tags=["Inference"], summary="Vision Helper")
async def vision(
    model: str = Form(..., description="VLM Model ID"),
    prompt: str = Form(..., description="Text prompt about the image"),
    file: UploadFile = File(..., description="Image file"),
    options: str = Form("{}", description="JSON string of options"),
):
    """
    **Image-to-Text Helper.**

    Convenience endpoint handling file upload + base64 conversion + generation in one step.
    Useful for clients that don't want to handle base64 encoding manually.
    """
    temp_path = TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with open(temp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        return engine.generate(model, prompt, [img_b64], options=json.loads(options))
    finally:
        temp_path.unlink(missing_ok=True)


# Legacy
@app.get("/available-models", tags=["Legacy"], include_in_schema=False)
def list_available():
    """
    List all models configured in models.json.

    This is a legacy endpoint providing raw access to the server's
    allow-list configuration.

    Returns
    -------
    dict
        A dictionary mapping model IDs to their full configuration objects.
    """
    return {k: v.model_dump() for k, v in manager.get_allowed_models().items()}


@app.get("/models", tags=["Legacy"], include_in_schema=False)
def list_downloaded():
    """
    List all models currently present in the local registry.

    This is a legacy endpoint providing raw access to the internal
    registry state, including download status and local paths.

    Returns
    -------
    dict
        A dictionary mapping model IDs to their registry status objects.
    """
    return {k: v.model_dump() for k, v in manager.get_registry().items()}