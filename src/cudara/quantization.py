"""
Quantization System for Cudara
==============================
Implements 'Dynamic' quantization strategies using BitsAndBytes.
Configuration is strictly driven by input arguments and 'models.json',
avoiding hardcoded model-specific logic in Python.
"""

import gc
import importlib.util
import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class ModelCategory(str, Enum):
    """
    Generic model categories.
    Specific requirements (e.g., Qwen vs Llama skips) must be defined
    via 'skip_modules' in the quantization config, not here.
    """

    TEXT_LLM = "text_llm"
    VLM = "vlm"
    ASR = "asr"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


class QuantizationMethod(str, Enum):
    """Supported quantization backends."""

    BITSANDBYTES = "bitsandbytes"
    NONE = "none"


@dataclass
class QuantizationConfig:
    """
    Quantization settings for a model.
    """

    method: QuantizationMethod = QuantizationMethod.BITSANDBYTES
    bits: int = 4
    skip_modules: List[str] = field(default_factory=list)
    use_double_quant: bool = True
    quant_type: str = "nf4"


# Default Profiles
# These are fallback defaults. Specific models should define 'skip_modules'
# in models.json if they deviate from these defaults.
QUANTIZATION_PROFILES: Dict[ModelCategory, QuantizationConfig] = {
    ModelCategory.TEXT_LLM: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=["lm_head", "embed_tokens"],
        use_double_quant=True,
    ),
    ModelCategory.VLM: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        # Safe defaults for most VLMs.
        # For Llama 3.2 Vision, add 'cross_attn' to skip_modules in models.json.
        # For Qwen-VL, add 'merger'/'vision_tower' to skip_modules in models.json.
        skip_modules=["lm_head", "embed_tokens", "vision_model", "vision_tower"],
        use_double_quant=True,
    ),
    ModelCategory.ASR: QuantizationConfig(method=QuantizationMethod.NONE),
    ModelCategory.EMBEDDING: QuantizationConfig(method=QuantizationMethod.NONE),
    ModelCategory.UNKNOWN: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES, bits=4, skip_modules=["lm_head"]
    ),
}


def detect_model_category(task: Optional[str] = None) -> ModelCategory:
    """
    Detect generic model category based on task.
    Does NOT use model name sniffing.
    """
    if not task:
        return ModelCategory.TEXT_LLM

    if task == "automatic-speech-recognition":
        return ModelCategory.ASR

    if task == "feature-extraction":
        return ModelCategory.EMBEDDING

    if task == "image-to-text":
        return ModelCategory.VLM

    return ModelCategory.TEXT_LLM


class BitsAndBytesQuantizer:
    """BitsAndBytes quantizer wrapper implementing NF4 and Double Quantization."""

    @staticmethod
    def is_available() -> bool:
        """Check if bitsandbytes is installed."""
        return importlib.util.find_spec("bitsandbytes") is not None

    @staticmethod
    def quantize(
        model_path: Path,
        output_path: Path,
        config: QuantizationConfig,
        trust_remote_code: bool = False,
        architecture: str = "AutoModelForCausalLM",
    ) -> Dict[str, Any]:
        """
        Quantize model using BitsAndBytes with NF4 and BFloat16 computation.
        """
        logger.info(f"Quantizing with BnB {config.bits}-bit (NF4)...")

        # Use BFloat16 if available for stability
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if config.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=config.use_double_quant,
                llm_int8_skip_modules=config.skip_modules if config.skip_modules else None,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=config.skip_modules if config.skip_modules else None,
            )

        model_cls = getattr(transformers, architecture, AutoModelForCausalLM)

        # Load into memory (RAM -> VRAM during quantization)
        with torch.no_grad():
            model = model_cls.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=compute_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path), safe_serialization=True)

        # Save processor/tokenizer artifacts
        try:
            processor = load_processor_with_fallback(model_path, trust_remote_code)
            processor.save_pretrained(str(output_path))
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path), trust_remote_code=trust_remote_code
                )
                tokenizer.save_pretrained(str(output_path))
            except Exception:
                logger.warning("Could not save tokenizer/processor artifacts.")

        return {
            "method": "bitsandbytes",
            "bits": config.bits,
            "quant_type": "nf4",
            "double_quant": config.use_double_quant,
            "skipped_modules": config.skip_modules,
        }


class ModelQuantizer:
    """Main facade for the quantization system."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._bnb_available = BitsAndBytesQuantizer.is_available()

        if self._bnb_available:
            logger.info("✓ BitsAndBytes available")
        else:
            logger.warning("BitsAndBytes not available - quantization disabled")

    def prequantize_model(
        self,
        model_id: str,
        source_path: Path,
        target_path: Path,
        task: str = "text-generation",
        trust_remote_code: bool = False,
        architecture: str = "AutoModelForCausalLM",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-quantize a model.
        Prioritizes settings from 'custom_config' (from models.json) over defaults.
        """
        category = detect_model_category(task=task)

        # Start with default profile
        profile = QUANTIZATION_PROFILES.get(category, QUANTIZATION_PROFILES[ModelCategory.UNKNOWN])

        # Apply overrides from models.json
        if custom_config:
            if "category" in custom_config:
                # If explicit category provided in JSON (e.g. 'vlm_llama' mapped to VLM)
                # currently we only have generic VLM, so we rely on skip_modules mostly.
                pass

            if "skip_modules" in custom_config and custom_config["skip_modules"]:
                profile.skip_modules = custom_config["skip_modules"]

            if "bits" in custom_config:
                profile.bits = custom_config["bits"]

        logger.info(f"Strategy: {category.value} | Skip: {profile.skip_modules}")

        # Handle 'NONE' or missing backend
        if profile.method == QuantizationMethod.NONE:
            logger.info("Quantization disabled for this model type")
            if source_path != target_path:
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            return {"quantized": False, "category": category.value}

        if not self._bnb_available:
            logger.warning("No quantization backend - copying unquantized")
            if source_path != target_path:
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            return {"quantized": False, "category": category.value, "error": "No backend"}

        try:
            result = BitsAndBytesQuantizer.quantize(
                model_path=source_path,
                output_path=target_path,
                config=profile,
                trust_remote_code=trust_remote_code,
                architecture=architecture,
            )

            self._copy_additional_files(source_path, target_path, trust_remote_code)

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if source_path != target_path and source_path.exists():
                shutil.rmtree(source_path, ignore_errors=True)

            logger.info("✓ Quantization complete")
            return {"quantized": True, "category": category.value, **result}

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            if source_path != target_path:
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            return {"quantized": False, "category": category.value, "error": str(e)}

    def _copy_additional_files(self, source: Path, target: Path, trust_remote_code: bool):
        """Copy config files needed for the model."""
        files = [
            "generation_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "video_preprocessor.json",
            "chat_template.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "added_tokens.json",
        ]

        for f in files:
            src = source / f
            if src.exists() and not (target / f).exists():
                shutil.copy2(src, target / f)

        if trust_remote_code:
            for py in source.glob("*.py"):
                if not (target / py.name).exists():
                    shutil.copy2(py, target / py.name)

        ensure_video_preprocessor_json(target)


# =============================================================================
# PROCESSOR UTILITIES
# =============================================================================

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


def ensure_video_preprocessor_json(model_path: Path) -> bool:
    """
    Ensure video configuration exists.
    Checks config content rather than model name to decide logic.
    """
    video_config = model_path / "video_preprocessor.json"
    preproc_config = model_path / "preprocessor_config.json"

    if video_config.exists():
        return True

    if not preproc_config.exists():
        return False

    try:
        with open(preproc_config, "r") as f:
            config = json.load(f)

        # Check for Qwen-specific class in the config file, not the model name
        proc_class = config.get("processor_class", "").lower()
        if "qwen2" in proc_class and "video" not in proc_class:
            with open(video_config, "w") as f:
                json.dump(
                    {
                        "processor_class": "Qwen2VLVideoProcessor",
                        "min_pixels": config.get("min_pixels", 3136),
                        "max_pixels": config.get("max_pixels", 12845056),
                        "patch_size": config.get("patch_size", 14),
                        "temporal_patch_size": config.get("temporal_patch_size", 2),
                        "merge_size": config.get("merge_size", 2),
                    },
                    f,
                    indent=2,
                )
            return True
    except Exception:
        pass
    return False


def sync_token_ids_with_tokenizer(model, tokenizer) -> None:
    """Sync model config token IDs with tokenizer."""
    if not hasattr(model, "config"):
        return

    tok = getattr(tokenizer, "tokenizer", tokenizer)

    mapping = {
        "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>",
        "vision_start_token_id": "<|vision_start|>",
        "vision_end_token_id": "<|vision_end|>",
    }

    for attr, token in mapping.items():
        if token not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [token]})

        real_id = tok.convert_tokens_to_ids(token)

        if hasattr(model.config, attr):
            setattr(model.config, attr, real_id)


def load_processor_with_fallback(model_path: Path, trust_remote_code: bool = False):
    """
    Load processor with generic fallbacks.
    Does not rely on hardcoded model names in the logic flow.
    """
    ensure_video_preprocessor_json(model_path)

    # 1. Try Standard AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
        _ensure_chat_template(processor)
        return processor
    except (TypeError, ValueError):
        logger.debug("AutoProcessor failed, attempting component load.")

    # 2. Try Component Loading (Common for VLMs)
    try:
        from transformers import AutoImageProcessor, AutoTokenizer

        # Load components separately
        image_proc = AutoImageProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )

        # Try to find a specialized processor dynamically
        # This part requires specific imports but we guard them
        try:
            from transformers import Qwen2VLProcessor, Qwen2VLVideoProcessor

            try:
                video_proc = Qwen2VLVideoProcessor.from_pretrained(str(model_path))
            except Exception:
                video_proc = None

            processor = Qwen2VLProcessor(
                image_processor=image_proc, tokenizer=tokenizer, video_processor=video_proc
            )
            _ensure_chat_template(processor)
            return processor
        except ImportError:
            pass

        # If no specialized processor class found, return wrapper
        return _create_wrapper(image_proc, tokenizer)

    except Exception as e:
        logger.warning(f"Component loading failed: {e}. Falling back to text-only.")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
        return _create_wrapper(None, tok)


def _ensure_chat_template(proc):
    """Inject template if missing (mostly for Qwen-like models)."""
    tok = getattr(proc, "tokenizer", proc)
    if not hasattr(tok, "chat_template") or not tok.chat_template:
        tok.chat_template = QWEN_VL_TEMPLATE


def _create_wrapper(image_processor, tokenizer):
    """Generic wrapper for separated tokenizer/image_processor."""

    class Wrapper:
        def __init__(self, ip, tok):
            self.image_processor = ip
            self.tokenizer = tok
            self.chat_template = getattr(tok, "chat_template", QWEN_VL_TEMPLATE)

        def __call__(self, text=None, images=None, **kwargs):
            return self.tokenizer(text, **kwargs)

        def batch_decode(self, *args, **kwargs):
            return self.tokenizer.batch_decode(*args, **kwargs)

        def save_pretrained(self, path):
            self.tokenizer.save_pretrained(path)
            if self.image_processor:
                self.image_processor.save_pretrained(path)

    return Wrapper(image_processor, tokenizer)


__all__ = [
    "ModelQuantizer",
    "ModelCategory",
    "QuantizationMethod",
    "QuantizationConfig",
    "QUANTIZATION_PROFILES",
    "BitsAndBytesQuantizer",
    "detect_model_category",
    "sync_token_ids_with_tokenizer",
    "ensure_video_preprocessor_json",
    "load_processor_with_fallback",
]
