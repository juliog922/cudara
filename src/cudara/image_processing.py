"""
Adaptive Image Processing for VLM Inference
============================================
VRAM-aware resolution calculation with GPU presets.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class VRAMProfile:
    """GPU VRAM configuration for adaptive processing."""
    total_vram_gb: float
    available_vram_gb: float
    model_vram_gb: float
    
    max_pixels: int = 0
    optimal_pixels: int = 0
    min_pixels: int = 200704  # 256 * 28 * 28
    resolution_tiers: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self._calculate_optimal_settings()
    
    def _calculate_optimal_settings(self):
        """Calculate optimal image settings based on available VRAM."""
        buffer_for_inference = self.available_vram_gb * 0.4
        usable_for_vision = max(0.5, self.available_vram_gb - buffer_for_inference)
        pixels_per_gb = 2_500_000
        
        estimated_max_pixels = int(usable_for_vision * pixels_per_gb)
        self.max_pixels = min(max(estimated_max_pixels, self.min_pixels), 3_000_000)
        self.optimal_pixels = int(self.max_pixels * 0.65)
        self._build_resolution_tiers()
    
    def _build_resolution_tiers(self):
        """Build resolution tiers for OOM recovery."""
        tiers = []
        current = self.optimal_pixels
        
        while current > self.min_pixels and len(tiers) < 5:
            tiers.append(current)
            current = int(current * 0.5)
        
        tiers.append(self.min_pixels)
        self.resolution_tiers = tiers


@dataclass 
class ImageProcessingConfig:
    """Configuration for adaptive image processing."""
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 200704
    max_pixels: int = 802816
    optimal_pixels: int = 602112
    enable_progressive_fallback: bool = True
    max_retry_attempts: int = 3
    downscale_factor: float = 0.5
    preserve_aspect_ratio: bool = True
    use_lanczos_resize: bool = True
    pixels_per_token: int = 784


class AdaptiveImageProcessor:
    """Optimized adaptive image processor for VLM inference."""
    
    def __init__(
        self,
        config: Optional[ImageProcessingConfig] = None,
        vram_profile: Optional[VRAMProfile] = None,
    ):
        self.config = config or ImageProcessingConfig()
        self.vram_profile = vram_profile
        self._current_tier_index = 0
        
        if vram_profile:
            self.config.min_pixels = vram_profile.min_pixels
            self.config.max_pixels = vram_profile.max_pixels
            self.config.optimal_pixels = vram_profile.optimal_pixels
    
    @classmethod
    def from_gpu_info(cls, model_vram_gb: float = 0) -> "AdaptiveImageProcessor":
        """Create processor with automatic GPU detection."""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            total_gb = total / (1024**3)
            available_gb = free / (1024**3)
            
            vram_profile = VRAMProfile(
                total_vram_gb=total_gb,
                available_vram_gb=available_gb,
                model_vram_gb=model_vram_gb,
            )
            
            logger.info(f"GPU: {total_gb:.1f}GB total, {available_gb:.1f}GB free -> max {vram_profile.max_pixels//1000}K px")
            return cls(vram_profile=vram_profile)
        
        return cls()
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_resize_dims(
        original_width: int, 
        original_height: int, 
        max_pixels: int,
        min_pixels: int,
        patch_factor: int = 28
    ) -> Tuple[int, int]:
        """Cached dimension calculation."""
        original_pixels = original_width * original_height
        
        if min_pixels <= original_pixels <= max_pixels:
            new_width = max(patch_factor, round(original_width / patch_factor) * patch_factor)
            new_height = max(patch_factor, round(original_height / patch_factor) * patch_factor)
            return new_width, new_height
        
        if original_pixels > max_pixels:
            scale = math.sqrt(max_pixels / original_pixels)
        else:
            scale = math.sqrt(min_pixels / original_pixels)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        new_width = max(patch_factor, round(new_width / patch_factor) * patch_factor)
        new_height = max(patch_factor, round(new_height / patch_factor) * patch_factor)
        
        while new_width * new_height > max_pixels:
            if new_width > new_height:
                new_width = max(patch_factor, new_width - patch_factor)
            else:
                new_height = max(patch_factor, new_height - patch_factor)
        
        return new_width, new_height
    
    def estimate_tokens(self, width: int, height: int) -> int:
        """Estimate visual token count."""
        pixels = width * height
        return max(1, pixels // self.config.pixels_per_token)
    
    def estimate_vram_usage(self, width: int, height: int) -> float:
        """Estimate VRAM usage in GB."""
        tokens = self.estimate_tokens(width, height)
        return tokens * 0.4 / 1024
    
    def smart_resize(
        self,
        image: Image.Image,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> Image.Image:
        """Fast resize with aspect ratio preservation."""
        max_pixels = max_pixels or self.config.optimal_pixels
        min_pixels = min_pixels or self.config.min_pixels
        
        original_width, original_height = image.size
        patch_factor = self.config.patch_size * self.config.merge_size
        
        new_width, new_height = self._calculate_resize_dims(
            original_width, original_height, max_pixels, min_pixels, patch_factor
        )
        
        if (new_width, new_height) == (original_width, original_height):
            return image
        
        resample = Image.Resampling.LANCZOS if self.config.use_lanczos_resize else Image.Resampling.BILINEAR
        return image.resize((new_width, new_height), resample=resample)
    
    def prepare_for_inference(
        self,
        image: Image.Image,
        tier_index: int = 0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Prepare image for VLM inference with metadata."""
        original_size = image.size
        original_pixels = original_size[0] * original_size[1]
        
        if self.vram_profile and self.vram_profile.resolution_tiers:
            tier_index = min(tier_index, len(self.vram_profile.resolution_tiers) - 1)
            max_pixels = self.vram_profile.resolution_tiers[tier_index]
        else:
            max_pixels = int(self.config.optimal_pixels * (self.config.downscale_factor ** tier_index))
            max_pixels = max(max_pixels, self.config.min_pixels)
        
        processed = self.smart_resize(image, max_pixels=max_pixels)
        new_size = processed.size
        new_pixels = new_size[0] * new_size[1]
        estimated_tokens = self.estimate_tokens(new_size[0], new_size[1])
        
        metadata = {
            "original_size": original_size,
            "original_pixels": original_pixels,
            "processed_size": new_size,
            "processed_pixels": new_pixels,
            "estimated_tokens": estimated_tokens,
            "tier_index": tier_index,
            "max_pixels_used": max_pixels,
            "scale_factor": new_pixels / original_pixels if original_pixels > 0 else 1.0,
        }
        
        logger.debug(f"Image: {new_size[0]}x{new_size[1]} -> {estimated_tokens} tokens")
        return processed, metadata
    
    def reset_tier(self):
        self._current_tier_index = 0
    
    def next_tier(self) -> bool:
        max_tiers = len(self.vram_profile.resolution_tiers) if self.vram_profile else self.config.max_retry_attempts
        if self._current_tier_index < max_tiers - 1:
            self._current_tier_index += 1
            return True
        return False
    
    @property
    def current_tier(self) -> int:
        return self._current_tier_index


def create_processor_with_dynamic_resolution(
    processor,
    vram_gb: float,
    model_loaded_gb: float = 0,
) -> Tuple[Any, ImageProcessingConfig]:
    """Configure HuggingFace processor with VRAM-aware resolution."""
    available_gb = vram_gb - model_loaded_gb
    
    profile = VRAMProfile(
        total_vram_gb=vram_gb,
        available_vram_gb=available_gb,
        model_vram_gb=model_loaded_gb,
    )
    
    config = ImageProcessingConfig(
        min_pixels=profile.min_pixels,
        max_pixels=profile.max_pixels,
        optimal_pixels=profile.optimal_pixels,
    )
    
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        if hasattr(ip, 'min_pixels'):
            ip.min_pixels = config.min_pixels
        if hasattr(ip, 'max_pixels'):
            ip.max_pixels = config.optimal_pixels
    
    return processor, config


# =============================================================================
# VRAM PRESETS - Common GPU configurations
# =============================================================================
VRAM_PRESETS = {
    # Consumer GPUs - GeForce RTX 30 Series
    "rtx_3060_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=7, model_vram_gb=3.5),
    "rtx_3070_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=4.5, model_vram_gb=3),
    "rtx_3080_10gb": VRAMProfile(total_vram_gb=10, available_vram_gb=6, model_vram_gb=3),
    "rtx_3080_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=8, model_vram_gb=3),
    "rtx_3090_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=16, model_vram_gb=4),
    "rtx_3090ti_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=17, model_vram_gb=4),
    
    # Consumer GPUs - GeForce RTX 40 Series
    "rtx_4060_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=5, model_vram_gb=3),
    "rtx_4060ti_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=5.5, model_vram_gb=3),
    "rtx_4060ti_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=11, model_vram_gb=4),
    "rtx_4070_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=8, model_vram_gb=3),
    "rtx_4070ti_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=8.5, model_vram_gb=3),
    "rtx_4070ti_super_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=11, model_vram_gb=4),
    "rtx_4080_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=11, model_vram_gb=4),
    "rtx_4080_super_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=12, model_vram_gb=4),
    "rtx_4090_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    
    # Consumer GPUs - GeForce RTX 50 Series (Blackwell)
    "rtx_5070_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=8, model_vram_gb=3),
    "rtx_5070ti_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=12, model_vram_gb=4),
    "rtx_5080_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=12, model_vram_gb=4),
    "rtx_5090_32gb": VRAMProfile(total_vram_gb=32, available_vram_gb=24, model_vram_gb=5),
    
    # Professional GPUs - NVIDIA RTX (Workstation)
    "rtx_a4000_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=12, model_vram_gb=4),
    "rtx_a5000_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    "rtx_a6000_48gb": VRAMProfile(total_vram_gb=48, available_vram_gb=38, model_vram_gb=6),
    "rtx_5000_ada_32gb": VRAMProfile(total_vram_gb=32, available_vram_gb=25, model_vram_gb=5),
    "rtx_6000_ada_48gb": VRAMProfile(total_vram_gb=48, available_vram_gb=38, model_vram_gb=6),
    
    # Data Center GPUs - NVIDIA A-Series (Ampere)
    "a10_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    "a10g_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    "a16_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=12, model_vram_gb=4),
    "a30_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    "a40_48gb": VRAMProfile(total_vram_gb=48, available_vram_gb=38, model_vram_gb=6),
    "a100_40gb": VRAMProfile(total_vram_gb=40, available_vram_gb=30, model_vram_gb=4),
    "a100_80gb": VRAMProfile(total_vram_gb=80, available_vram_gb=65, model_vram_gb=5),
    
    # Data Center GPUs - NVIDIA L-Series (Ada Lovelace)
    "l4_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4),
    "l40_48gb": VRAMProfile(total_vram_gb=48, available_vram_gb=38, model_vram_gb=6),
    "l40s_48gb": VRAMProfile(total_vram_gb=48, available_vram_gb=40, model_vram_gb=6),
    
    # Data Center GPUs - NVIDIA H-Series (Hopper)
    "h100_80gb": VRAMProfile(total_vram_gb=80, available_vram_gb=65, model_vram_gb=5),
    "h100_pcie_80gb": VRAMProfile(total_vram_gb=80, available_vram_gb=65, model_vram_gb=5),
    "h100_sxm_80gb": VRAMProfile(total_vram_gb=80, available_vram_gb=68, model_vram_gb=5),
    "h100_nvl_94gb": VRAMProfile(total_vram_gb=94, available_vram_gb=80, model_vram_gb=5),
    "h200_141gb": VRAMProfile(total_vram_gb=141, available_vram_gb=120, model_vram_gb=6),
    
    # Data Center GPUs - NVIDIA B-Series (Blackwell)
    "b100_192gb": VRAMProfile(total_vram_gb=192, available_vram_gb=160, model_vram_gb=8),
    "b200_192gb": VRAMProfile(total_vram_gb=192, available_vram_gb=165, model_vram_gb=8),
    "gb200_384gb": VRAMProfile(total_vram_gb=384, available_vram_gb=320, model_vram_gb=10),
    
    # Laptop GPUs (conservative estimates)
    "rtx_3060_laptop_6gb": VRAMProfile(total_vram_gb=6, available_vram_gb=3.5, model_vram_gb=2.5),
    "rtx_3070_laptop_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=5, model_vram_gb=3),
    "rtx_3080_laptop_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=11, model_vram_gb=4),
    "rtx_4060_laptop_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=5, model_vram_gb=3),
    "rtx_4070_laptop_8gb": VRAMProfile(total_vram_gb=8, available_vram_gb=5.5, model_vram_gb=3),
    "rtx_4080_laptop_12gb": VRAMProfile(total_vram_gb=12, available_vram_gb=8, model_vram_gb=3.5),
    "rtx_4090_laptop_16gb": VRAMProfile(total_vram_gb=16, available_vram_gb=11, model_vram_gb=4),
    "rtx_5090_laptop_24gb": VRAMProfile(total_vram_gb=24, available_vram_gb=17, model_vram_gb=5),
}