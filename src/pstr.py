from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import mmcv
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

CONFIG_FILE = str(Path.cwd() / "configs" / "pstr" / "tdbaseline.py")
WEIGHT_FILE = Path().home() / "models" "pstr_resnet_cuhk" / "pstr_r50_cuhk.pth"

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
        weight_file: Path = WEIGHT_FILE,
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
