"""
Quantization System for Cudara
==============================
Uses BitsAndBytes for quantization - no calibration needed, works out of the box.

Model Categories:
- text_llm_small: <4B params - use 4-bit with careful skip_modules
- text_llm_medium: 4B-13B params - optimal for 4-bit
- text_llm_large: >13B params - very robust to quantization
- vlm_small/medium/large: Vision-Language Models
- asr: Speech recognition - NO quantization
- embedding: Embeddings - NO quantization
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
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class ModelCategory(str, Enum):
    """
    Categorization of models for applying optimal quantization profiles.
    """

    TEXT_LLM_SMALL = "text_llm_small"
    TEXT_LLM_MEDIUM = "text_llm_medium"
    TEXT_LLM_LARGE = "text_llm_large"
    VLM_SMALL = "vlm_small"
    VLM_MEDIUM = "vlm_medium"
    VLM_LARGE = "vlm_large"
    ASR = "asr"
    EMBEDDING = "embedding"


class QuantizationMethod(str, Enum):
    """Supported quantization backends."""

    BITSANDBYTES = "bitsandbytes"
    NONE = "none"


@dataclass
class QuantizationConfig:
    """
    Quantization settings for a model.

    Attributes
    ----------
    method : QuantizationMethod
        Backend to use.
    bits : int
        Target bit depth (4 or 8).
    skip_modules : List[str]
        Modules to exclude from quantization for stability.
    use_double_quant : bool
        Whether to use nested quantization (BnB feature).
    """

    method: QuantizationMethod = QuantizationMethod.BITSANDBYTES
    bits: int = 4
    skip_modules: List[str] = field(default_factory=list)
    use_double_quant: bool = True


# Profiles optimized for different model categories
QUANTIZATION_PROFILES: Dict[ModelCategory, QuantizationConfig] = {
    ModelCategory.TEXT_LLM_SMALL: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=["lm_head", "embed_tokens"],
        use_double_quant=True,
    ),
    ModelCategory.TEXT_LLM_MEDIUM: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=["lm_head"],
        use_double_quant=True,
    ),
    ModelCategory.TEXT_LLM_LARGE: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=["lm_head"],
        use_double_quant=True,
    ),
    ModelCategory.VLM_SMALL: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=[
            "lm_head",
            "visual",
            "vision_tower",
            "vision_model",
            "multi_modal_projector",
            "merger",
            "patch_embed",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
        ],
        use_double_quant=True,
    ),
    ModelCategory.VLM_MEDIUM: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=[
            "lm_head",
            "visual",
            "vision_tower",
            "vision_model",
            "multi_modal_projector",
            "merger",
            "model.layers.0",
            "model.layers.1",
        ],
        use_double_quant=True,
    ),
    ModelCategory.VLM_LARGE: QuantizationConfig(
        method=QuantizationMethod.BITSANDBYTES,
        bits=4,
        skip_modules=["lm_head", "visual", "vision_tower", "multi_modal_projector"],
        use_double_quant=True,
    ),
    ModelCategory.ASR: QuantizationConfig(method=QuantizationMethod.NONE),
    ModelCategory.EMBEDDING: QuantizationConfig(method=QuantizationMethod.NONE),
}


def estimate_model_size(model_id: str, config: Optional[AutoConfig] = None) -> float:
    """
    Estimate model size in billions of parameters.

    Parameters
    ----------
    model_id : str
        Model identifier string.
    config : AutoConfig, optional
        Loaded model configuration.

    Returns
    -------
    float
        Estimated size in Billions (e.g., 7.0 for 7B).
    """
    model_id_lower = model_id.lower()

    patterns = {
        "0.5b": 0.5,
        "1b": 1,
        "1.5b": 1.5,
        "1.7b": 1.7,
        "2b": 2,
        "3b": 3,
        "4b": 4,
        "7b": 7,
        "8b": 8,
        "9b": 9,
        "11b": 11,
        "12b": 12,
        "13b": 13,
        "14b": 14,
        "32b": 32,
        "34b": 34,
        "70b": 70,
        "72b": 72,
    }

    for pattern, size in patterns.items():
        if pattern in model_id_lower:
            return size

    if config:
        try:
            hidden = getattr(config, "hidden_size", 4096)
            layers = getattr(config, "num_hidden_layers", 32)
            vocab = getattr(config, "vocab_size", 32000)
            return (hidden * hidden * 4 * layers + hidden * vocab * 2) / 1e9
        except Exception:
            pass

    return 7.0


def detect_model_category(
    model_id: str, task: Optional[str] = None, config: Optional[AutoConfig] = None
) -> ModelCategory:
    """
    Detect model category from ID and task for profiling.

    Parameters
    ----------
    model_id : str
        Model name.
    task : str, optional
        HF task type.
    config : AutoConfig, optional
        Model configuration.

    Returns
    -------
    ModelCategory
        Detected category.
    """
    model_id_lower = model_id.lower()

    if task == "automatic-speech-recognition" or "whisper" in model_id_lower:
        return ModelCategory.ASR

    embedding_hints = ["embedding", "minilm", "bge", "e5", "sentence-transformer", "gte"]
    if task == "feature-extraction" or any(h in model_id_lower for h in embedding_hints):
        return ModelCategory.EMBEDDING

    vlm_hints = ["vl", "vision", "vlm", "ocr", "pixtral", "llava", "cogvlm", "internvl", "molmo"]
    is_vlm = any(h in model_id_lower for h in vlm_hints)

    size = estimate_model_size(model_id, config)

    if is_vlm:
        if size < 4:
            return ModelCategory.VLM_SMALL
        elif size < 15:
            return ModelCategory.VLM_MEDIUM
        return ModelCategory.VLM_LARGE
    else:
        if size < 4:
            return ModelCategory.TEXT_LLM_SMALL
        elif size < 13:
            return ModelCategory.TEXT_LLM_MEDIUM
        return ModelCategory.TEXT_LLM_LARGE


class BitsAndBytesQuantizer:
    """BitsAndBytes quantizer - works without calibration data."""

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
        Quantize model using BitsAndBytes.

        Parameters
        ----------
        model_path : Path
            Input model directory.
        output_path : Path
            Output directory for quantized weights.
        config : QuantizationConfig
            Quantization parameters.
        trust_remote_code : bool
            Allow custom code execution.
        architecture : str
            Model class name (e.g. AutoModelForCausalLM).

        Returns
        -------
        Dict
            Metadata about the operation.
        """
        logger.info(f"Quantizing with BnB {config.bits}-bit...")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if config.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
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

        model = model_cls.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path), safe_serialization=True)

        # Save processor/tokenizer
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
                pass

        return {
            "method": "bitsandbytes",
            "bits": config.bits,
            "quant_type": "nf4" if config.bits == 4 else "int8",
            "double_quant": config.use_double_quant,
        }


class ModelQuantizer:
    """Main quantizer facade class."""

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
    ) -> Dict[str, Any]:
        """
        Pre-quantize a model with optimal settings based on its category.

        Parameters
        ----------
        model_id : str
            Model identifier.
        source_path : Path
            Path to downloaded raw model.
        target_path : Path
            Path to save quantized model.
        task : str
            HF task type.
        trust_remote_code : bool
            Allow custom code.
        architecture : str
            Model class name.

        Returns
        -------
        Dict
            Result metadata including 'quantized' boolean.
        """
        try:
            config = AutoConfig.from_pretrained(
                str(source_path), trust_remote_code=trust_remote_code
            )
        except Exception:
            config = None

        category = detect_model_category(model_id, task=task, config=config)
        profile = QUANTIZATION_PROFILES.get(
            category, QUANTIZATION_PROFILES[ModelCategory.TEXT_LLM_MEDIUM]
        )

        logger.info(f"Category: {category.value}, Profile: {profile.bits}-bit")

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

QWEN_2_VL_CHAT_TEMPLATE = (
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
    Create video_preprocessor.json for Qwen-VL models if missing.

    Parameters
    ----------
    model_path : Path
        Directory containing model files.

    Returns
    -------
    bool
        True if file exists or was created.
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

        if "qwen2" in config.get("processor_class", "").lower():
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
    """
    Sync model config token IDs with tokenizer to fix Qwen/Vision issues.

    Parameters
    ----------
    model : AutoModel
        Loaded model instance.
    tokenizer : AutoTokenizer
        Loaded tokenizer instance.
    """
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
        if hasattr(tokenizer, attr):
            setattr(tokenizer, attr, real_id)


def load_processor_with_fallback(model_path: Path, trust_remote_code: bool = False):
    """
    Load processor with robust fallback handling for VLMs.

    Parameters
    ----------
    model_path : Path
        Directory containing model files.
    trust_remote_code : bool
        Allow custom code execution.

    Returns
    -------
    Any
        Processor instance or wrapper.
    """
    ensure_video_preprocessor_json(model_path)

    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
        _ensure_chat_template(processor)
        return processor
    except (TypeError, ValueError):
        pass

    try:
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen2VLProcessor

        image_proc = AutoImageProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )

        try:
            from transformers import Qwen2VLVideoProcessor

            video_proc = Qwen2VLVideoProcessor.from_pretrained(str(model_path))
        except Exception:
            video_proc = None

        if video_proc:
            processor = Qwen2VLProcessor(
                image_processor=image_proc, tokenizer=tokenizer, video_processor=video_proc
            )
        else:
            try:
                processor = Qwen2VLProcessor(image_processor=image_proc, tokenizer=tokenizer)
            except Exception:

                class MockVideo:
                    def __call__(self, *a, **k):
                        return {}

                processor = Qwen2VLProcessor(
                    image_processor=image_proc, tokenizer=tokenizer, video_processor=MockVideo()
                )

        _ensure_chat_template(processor)
        return processor

    except Exception as e:
        logger.warning(f"Using wrapper fallback: {e}")
        from transformers import AutoImageProcessor, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
        try:
            img = AutoImageProcessor.from_pretrained(
                str(model_path), trust_remote_code=trust_remote_code
            )
        except Exception:
            img = None
        return _create_wrapper(img, tok)


def _ensure_chat_template(proc):
    tok = getattr(proc, "tokenizer", proc)
    if not hasattr(tok, "chat_template") or not tok.chat_template:
        tok.chat_template = QWEN_2_VL_CHAT_TEMPLATE


def _create_wrapper(image_processor, tokenizer):
    class Wrapper:
        def __init__(self, ip, tok):
            self.image_processor = ip
            self.tokenizer = tok
            self.chat_template = QWEN_2_VL_CHAT_TEMPLATE
            for attr in ["pad_token", "pad_token_id", "eos_token", "eos_token_id"]:
                if hasattr(tok, attr):
                    setattr(self, attr, getattr(tok, attr))

        def __call__(self, text=None, images=None, **kwargs):
            from transformers import BatchFeature

            res = {}
            if text:
                res.update(self.tokenizer(text, **kwargs))
            if images and self.image_processor:
                try:
                    res.update(self.image_processor(images, return_tensors="pt"))
                except Exception:
                    pass
            return BatchFeature(res)

        def apply_chat_template(self, *args, **kwargs):
            return self.tokenizer.apply_chat_template(*args, **kwargs)

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
