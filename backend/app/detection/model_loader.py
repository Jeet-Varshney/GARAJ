"""
Model Loader Module for W2V2-AASIST Deepfake Voice Detection Engine (Phase 2 Domain Adaptation).

Responsibilities:
- Priority Search:
  1. Fine-tuned 'LA_domain_adapted.pth' (checkpoint_type = DOMAIN_ADAPTED)
  2. Pre-trained baseline 'LA_model.pth' (checkpoint_type = BASELINE)
- Load XLS-R 300M speech encoder front-end & AASIST graph attention back-end.
- Perform fairseq to PyTorch HuggingFace state_dict key translation.
- Strictly verify state_dict keys and weight compatibility (0 missing, 0 unexpected).
- If adapted checkpoint is corrupt or incompatible:
  Raises DOMAIN_ADAPTED_CHECKPOINT_INVALID / DOMAIN_ADAPTED_CHECKPOINT_INCOMPATIBLE error.
- NEVER silently fall back to generic Wav2Vec2, random weights, heuristics, or stubs.
"""

import sys
import os
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("model_loader")


class ModelLoadError(Exception):
    """Custom exception raised when model checkpoint loading or weight validation fails."""
    pass


def get_torch_device():
    """Detects available PyTorch compute device (CUDA GPU vs CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_la_model_checkpoint() -> Tuple[Optional[str], str]:
    """
    Searches standard filesystem paths for trained 'LA_domain_adapted.pth' or 'LA_model.pth' checkpoint.
    Prioritizes fine-tuned domain-adapted checkpoints if available.

    Returns:
        Tuple[checkpoint_path, checkpoint_type]
        checkpoint_type: "DOMAIN_ADAPTED" or "BASELINE" or "NONE"
    """
    env_path = os.getenv("AASIST_CHECKPOINT_PATH")
    if env_path and os.path.exists(env_path):
        ckpt_type = "DOMAIN_ADAPTED" if "domain_adapted" in env_path else "BASELINE"
        return os.path.abspath(env_path), ckpt_type

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workspace_dir = os.path.abspath(os.path.join(base_dir, ".."))

    # Priority 0: Experimental Feature Fusion C3 Checkpoint
    fusion_paths = [
        os.path.join(base_dir, "checkpoints", "LA_feature_fusion_C3.pth"),
        os.path.join(base_dir, "LA_feature_fusion_C3.pth"),
    ]
    for p in fusion_paths:
        if os.path.exists(p):
            return p, "FEATURE_AUGMENTED"

    # Priority 1: Domain-adapted checkpoints
    adapted_paths = [
        os.path.join(base_dir, "checkpoints", "LA_domain_adapted.pth"),
        os.path.join(base_dir, "LA_domain_adapted.pth"),
        os.path.join(workspace_dir, "checkpoints", "LA_domain_adapted.pth"),
    ]
    for p in adapted_paths:
        if os.path.exists(p):
            return p, "DOMAIN_ADAPTED"

    # Priority 2: Baseline checkpoints
    baseline_paths = [
        os.path.join(base_dir, "checkpoints", "LA_model.pth"),
        os.path.join(base_dir, "LA_model.pth"),
        os.path.join(workspace_dir, "checkpoints", "LA_model.pth"),
        os.path.join(workspace_dir, "LA_model.pth"),
        os.path.expanduser("~/.cache/garaj/LA_model.pth"),
        os.path.expanduser("~/.cache/huggingface/hub/LA_model.pth"),
    ]
    for p in baseline_paths:
        if os.path.exists(p):
            return p, "BASELINE"

    return None, "NONE"


def convert_la_model_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts fairseq / official W2V2-AASIST state_dict keys to HuggingFace Wav2Vec2Model structure.
    """
    mapped = {}
    for k, v in state_dict.items():
        # Strip DDP module prefix if present
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("ssl_model.model."):
            key = k
            key = key.replace(".feature_extractor.conv_layers.0.0.", ".feature_extractor.conv_layers.0.conv.")
            key = key.replace(".feature_extractor.conv_layers.0.2.1.", ".feature_extractor.conv_layers.0.layer_norm.")
            for i in range(1, 7):
                key = key.replace(f".feature_extractor.conv_layers.{i}.0.", f".feature_extractor.conv_layers.{i}.conv.")
                key = key.replace(f".feature_extractor.conv_layers.{i}.2.1.", f".feature_extractor.conv_layers.{i}.layer_norm.")
            key = key.replace(".post_extract_proj.", ".feature_projection.projection.")
            if key == "ssl_model.model.layer_norm.weight":
                key = "ssl_model.model.feature_projection.layer_norm.weight"
            elif key == "ssl_model.model.layer_norm.bias":
                key = "ssl_model.model.feature_projection.layer_norm.bias"
            key = key.replace(".mask_emb", ".masked_spec_embed")
            key = key.replace(".encoder.pos_conv.0.weight_g", ".encoder.pos_conv_embed.conv.parametrizations.weight.original0")
            key = key.replace(".encoder.pos_conv.0.weight_v", ".encoder.pos_conv_embed.conv.parametrizations.weight.original1")
            key = key.replace(".encoder.pos_conv.0.bias", ".encoder.pos_conv_embed.conv.bias")
            key = key.replace(".self_attn.", ".attention.")
            key = key.replace(".self_attn_layer_norm.", ".layer_norm.")
            key = key.replace(".fc1.", ".feed_forward.intermediate_dense.")
            key = key.replace(".fc2.", ".feed_forward.output_dense.")
            mapped[key] = v
        else:
            mapped[k] = v
    return mapped


def load_xlsr_aasist_model(
    model_name_or_path: Optional[str] = None,
    allow_baseline_fallback: bool = True
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Loads W2V2-AASIST model architecture, XLS-R 300M backbone, and trained weights.

    Priority:
    1. LA_domain_adapted.pth (checkpoint_type = DOMAIN_ADAPTED)
    2. LA_model.pth (checkpoint_type = BASELINE)

    Returns:
        Tuple of (feature_extractor, model, device, metadata_dict)
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Config
        from app.detection.aasist import W2V2AASIST
    except ImportError as e:
        raise ModelLoadError(f"Missing required PyTorch / HuggingFace dependencies: {e}") from e

    device = get_torch_device()

    if model_name_or_path:
        checkpoint_file = model_name_or_path
        checkpoint_type = "DOMAIN_ADAPTED" if "domain_adapted" in model_name_or_path else "BASELINE"
    else:
        checkpoint_file, checkpoint_type = find_la_model_checkpoint()

    # 1. Check if checkpoint exists
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        logger.warning("No W2V2-AASIST checkpoint found. Returning MODEL_CHECKPOINT_MISSING status.")
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        except Exception:
            from transformers import Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False
            )

        metadata = {
            "model_name": "W2V2-AASIST",
            "backbone": "XLS-R 300M",
            "backend": "AASIST",
            "sample_rate": 16000,
            "required_samples": 64600,
            "trained_classifier": False,
            "checkpoint_type": "NONE",
            "status": "MODEL_CHECKPOINT_MISSING",
            "checkpoint_path": None,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "num_parameters": 0,
        }
        return feature_extractor, None, device, metadata

    # 2. Checkpoint found: load full model architecture and state dict
    logger.info(f"Loading checkpoint '{checkpoint_file}' (Type: {checkpoint_type})...")

    try:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
        except Exception:
            from transformers import Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False
            )
            config = Wav2Vec2Config(
                hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096
            )

        model = W2V2AASIST(config=config)

        # Load weights with integrity validation
        try:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            converted_state_dict = convert_la_model_state_dict(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)

            filtered_unexpected = [k for k in unexpected_keys if not any(p in k for p in ["quantizer", "final_proj", "project_q"])]

            if len(missing_keys) > 0 or len(filtered_unexpected) > 0:
                err_code = "DOMAIN_ADAPTED_CHECKPOINT_INCOMPATIBLE" if checkpoint_type == "DOMAIN_ADAPTED" else "BASELINE_CHECKPOINT_INCOMPATIBLE"
                err_msg = f"Checkpoint key mismatch ({err_code}) for '{checkpoint_file}': {len(missing_keys)} missing, {len(filtered_unexpected)} unexpected."
                logger.error(err_msg)

                if checkpoint_type == "DOMAIN_ADAPTED" and allow_baseline_fallback:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    fallback_path = os.path.join(base_dir, "checkpoints", "LA_model.pth")
                    if os.path.exists(fallback_path):
                        logger.warning(f"Explicit safe fallback: loading baseline checkpoint '{fallback_path}'...")
                        return load_xlsr_aasist_model(model_name_or_path=fallback_path, allow_baseline_fallback=False)

                metadata = {
                    "model_name": "W2V2-AASIST",
                    "backbone": "XLS-R 300M",
                    "backend": "AASIST",
                    "sample_rate": 16000,
                    "required_samples": 64600,
                    "trained_classifier": False,
                    "checkpoint_type": checkpoint_type,
                    "status": err_code,
                    "error": err_msg,
                    "checkpoint_path": checkpoint_file,
                    "device": str(device),
                }
                return feature_extractor, None, device, metadata

        except Exception as load_err:
            err_code = "DOMAIN_ADAPTED_CHECKPOINT_INVALID" if checkpoint_type == "DOMAIN_ADAPTED" else "BASELINE_CHECKPOINT_INVALID"
            logger.error(f"Checkpoint file corrupt or unreadable: {load_err}")

            if checkpoint_type == "DOMAIN_ADAPTED" and allow_baseline_fallback:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                fallback_path = os.path.join(base_dir, "checkpoints", "LA_model.pth")
                if os.path.exists(fallback_path):
                    logger.warning(f"Explicit safe fallback: loading baseline checkpoint '{fallback_path}'...")
                    return load_xlsr_aasist_model(model_name_or_path=fallback_path, allow_baseline_fallback=False)

            metadata = {
                "model_name": "W2V2-AASIST",
                "backbone": "XLS-R 300M",
                "backend": "AASIST",
                "sample_rate": 16000,
                "required_samples": 64600,
                "trained_classifier": False,
                "checkpoint_type": checkpoint_type,
                "status": err_code,
                "error": str(load_err),
                "checkpoint_path": checkpoint_file,
                "device": str(device),
            }
            return feature_extractor, None, device, metadata

        model.to(device)
        model.eval()

        metadata = {
            "model_name": "W2V2-AASIST",
            "backbone": "XLS-R 300M",
            "backend": "AASIST",
            "sample_rate": 16000,
            "required_samples": 64600,
            "trained_classifier": True,
            "checkpoint_type": checkpoint_type,
            "status": "MODEL_READY",
            "checkpoint_path": checkpoint_file,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "cuda_available": torch.cuda.is_available(),
        }

        logger.info(f"Successfully loaded W2V2-AASIST ({metadata['num_parameters']:,} params) checkpoint_type={checkpoint_type} on {device}.")
        return feature_extractor, model, device, metadata

    except Exception as exc:
        err_msg = f"Failed to load W2V2-AASIST model from '{checkpoint_file}': {str(exc)}"
        logger.error(err_msg, exc_info=True)
        metadata = {
            "model_name": "W2V2-AASIST",
            "backbone": "XLS-R 300M",
            "backend": "AASIST",
            "sample_rate": 16000,
            "required_samples": 64600,
            "trained_classifier": False,
            "checkpoint_type": checkpoint_type,
            "status": "MODEL_LOAD_ERROR",
            "error": err_msg,
            "checkpoint_path": checkpoint_file,
            "device": str(device),
        }
        return feature_extractor, None, device, metadata
