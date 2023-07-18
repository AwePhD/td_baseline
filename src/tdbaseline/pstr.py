from pathlib import Path
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.models import build_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

CONFIG_FILE = str(Path.cwd() / "configs" / "pstr" / "tdbaseline.py")
WEIGHT_FILE = Path().home() / "models" / "pstr_resnet_cuhk" / "pstr_r50_cuhk.pth"

class PSTR:
    """
    A wrapper object of PSTR from MMDET.

    - It is initialized by a config file and the weight of the model.
    - It can run infer that outputs PSTR result for the whole dataset.
    """
    def _load_config(self, config_file: Path = CONFIG_FILE) -> mmcv.Config:
        cfg = mmcv.Config.fromfile(config_file)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
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
        load_checkpoint(model, str(weight_file), map_location='cpu')

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

    def infer(self) -> Dict[Path, np.ndarray]:
        with torch.no_grad():
            results = {
                Path(data['img_metas'][0].data[0][0]['filename']):
                    self.model(return_loss=False, rescale=True, **data)[0][0]
                for data in self.dataloader
            }
        return results
