from pathlib import Path
from typing import Dict

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import extract_int_from_str


class PSTR:
    """A wrapper object of PSTR from MMDET.

    - It is initialized by a config file and the weight of the model.
    - It can run infer that outputs PSTR result for the whole dataset.
    """

    def _load_config(
        self, config_file: Path, annotations_json: Path, root_folder: Path
    ) -> mmcv.Config:
        assert root_folder.exists() and annotations_json.exists()
        cfg = mmcv.Config.fromfile(config_file)
        cfg.data.test.ann_file = str(annotations_json)
        cfg.data.test.img_prefix = f"{root_folder}/test"
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        cfg.model.train_cfg = None
        print("Config in use:")
        print(cfg.pretty_text)

        return cfg

    def _load_dataloader(self) -> DataLoader:
        dataset = build_dataset(self.config.data.test)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False,
        )

        return dataloader

    def _load_model(self, weight_file: Path) -> MMDataParallel:
        # Get model and weights with boilerplate
        model = build_detector(
            self.config.model, test_cfg=self.config.get("test_cfg")
        ).eval()
        load_checkpoint(model, str(weight_file), map_location="cpu")

        model.CLASSES = self.dataloader.dataset.CLASSES  # type: ignore

        return MMDataParallel(model, device_ids=[0])

    def __init__(
        self,
        config_file: Path,
        annotations_json: Path,
        root_folder: Path,
        weight_file: Path,
    ) -> None:
        assert weight_file.exists()
        assert config_file.exists()

        self.config = self._load_config(
            config_file, annotations_json, root_folder
        )
        self.dataloader = self._load_dataloader()
        self.model = self._load_model(weight_file)

    def infer(self) -> Dict[int, np.ndarray]:
        results: Dict[int, np.ndarray] = {}
        for data in tqdm(self.dataloader):
            frame_id = extract_int_from_str(
                data["img_metas"][0].data[0][0]["filename"].split("/")[-1]
            )
            with torch.no_grad():
                results[frame_id] = self.model(
                    return_loss=False, rescale=True, **data
                )[0][0]
        return results
