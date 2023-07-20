from pathlib import Path

import torch
from irra.model.clip_model import CLIP, build_CLIP_from_openai_pretrained

WEIGHT_FILE = Path.home() / "models" / "clip_finetuned" / "clip_finetune.pth"
STRIDE_SIZE = 16
IMAGE_SIZE = (384, 128)


def load_clip(weight_file: Path = WEIGHT_FILE) -> CLIP:
    # Filter "base_mode." in front of params key.
    state_dict = {
        ".".join(key.split('.')[1:]): parameters
        for key, parameters in torch.load(weight_file)['model'].items()
    }

    model, _ = build_CLIP_from_openai_pretrained(
        "ViT-B/16", IMAGE_SIZE, STRIDE_SIZE)
    model.load_state_dict(state_dict)

    return model
