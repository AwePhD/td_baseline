from pathlib import Path
from typing import Dict, NamedTuple

from irra.model.clip_model import build_CLIP_from_openai_pretrained, CLIP

import torch
from torch.utils.data import DataLoader
import mmcv
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)


STRIDE_SIZE = 16
IMAGE_SIZE = (384, 128)

CONFIG_FILE = str(Path.cwd() / "configs" / "pstr" / "pstr_r50_24e_cuhk.py")
WEIGHT_FILE_CLIP = Path().home() / "models" / "clip_finetune.pth"
WEIGHT_FILE_PSTR = Path().home() / "models" / "pstr_r50_cuhk.pth"



class PSTR:
    """
    A wrapper object of PSTR from MMDET.

    - It is initialized by a config file and the weight of the model.
    - It can run infer that outputs PSTR result for the whole dataset.
    """
    def _load_config(self, config_file: Path) -> mmcv.Config:
        # Load configs file with boiler plate
        cfg = mmcv.Config.fromfile(str(config_file))
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
        cfg.model.train_cfg = None

        return cfg

    def _load_dataloader(self) -> DataLoader:
        dataset = build_dataset(self.config.data.test)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        return dataloader

    def _load_model(self, weight_file: Path) -> MMDataParallel:
        # Get model and weights with boilerplate
        model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg')).eval()
        fp16_cfg = self.config.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        model.CLASSES = self.dataloader.dataset.CLASSES

        return MMDataParallel(model, device_ids=[0])

    def __init__(
        self,
        config_file: Path = CONFIG_FILE,
        weight_file: Path = WEIGHT_FILE_PSTR,
    ) -> None:
        self.config = self._load_config(config_file)
        self.dataloader = self._load_dataloader()
        self.model = self._load_model(weight_file)

    def infer(self) -> Dict[Path, torch.Tensor]:
        with torch.no_grad():
            results = {
                Path(data['img_metas'][0].data[0][0]['filename']):
                    self.model(return_loss=False, rescale=True, **data)
                for data in self.dataloader
            }
        return results


def _create_model_config_from_state_dict(state_dict: Dict) -> Dict:
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith(
        "visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round(
        (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2]
        for k in state_dict
        if k.startswith(f"transformer.resblocks")
    ))

    return {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers,
        'vision_width': vision_width,
        'vision_patch_size': vision_patch_size,
        'context_length': context_length,
        'vocab_size': vocab_size,
        'transformer_width': transformer_width,
        'transformer_heads': transformer_heads,
        'transformer_layers': transformer_layers,
        'image_resolution': IMAGE_SIZE,
    }



def load_clip(weight_file: Path = WEIGHT_FILE_CLIP) -> CLIP:
    # Filter "base_mode." in front of params key.
    state_dict = {
        ".".join(key.split('.')[1:]): parameters
        for key, parameters in torch.load(weight_file)['model'].items()
    }

    model, _ = build_CLIP_from_openai_pretrained("ViT-B/16", IMAGE_SIZE, STRIDE_SIZE)

    return model.load_state_dict(state_dict)

