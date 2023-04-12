from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir
from mmseg.datasets import DATASETS

COCO_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "coco.py"
COCO_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "coco.yml"


class COCODataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, ann_dir=None, eval_split=None, **kwargs):
        self.names, self.colors = utils.dataset_cat_description(
            COCO_CONTEXT_CATS_PATH
        )
        self.n_cls = 91
        self.ignore_label = 255
        self.reduce_zero_label = False
        self.ann_dir = ann_dir
        self.eval_split = eval_split
        super().__init__(
            image_size, crop_size, split, COCO_CONTEXT_CONFIG_PATH, **kwargs
        )

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "coco"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path
            if self.ann_dir is not None:
                config.data.train.ann_dir = self.ann_dir
        elif self.split == "val":
            config.data.val.data_root = path
            if self.eval_split is not None:
                config.data.val.split = self.eval_split
        elif self.split == "test":
            raise ValueError("Test split is not valid for Pascal Context dataset")
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels
