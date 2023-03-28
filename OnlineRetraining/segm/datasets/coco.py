import os.path as osp

from mmseg.datasets.builder import DATASETS
# from .builder import DATASETS
from mmseg.datasets.custom import CustomDataset


# from .custom import CustomDataset


@DATASETS.register_module()
class COCODataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    # CLASSES = tuple([str(i) for i in range(91)])

    CLASSES = ('background',
               'person',
               'bicycle',
               'car',
               'motorcycle',
               'airplane',
               'bus',
               'train',
               'truck',
               'boat',
               'traffic light',
               'fire hydrant',
               'street sign',
               'stop sign',
               'parking meter',
               'bench',
               'bird',
               'cat',
               'dog',
               'horse',
               'sheep',
               'cow',
               'elephant',
               'bear',
               'zebra',
               'giraffe',
               'hat',
               'backpack',
               'umbrella',
               'shoe',
               'eye glasses',
               'handbag',
               'tie',
               'suitcase',
               'frisbee',
               'skis',
               'snowboard',
               'sports ball',
               'kite',
               'baseball bat',
               'baseball glove',
               'skateboard',
               'surfboard',
               'tennis racket',
               'bottle',
               'plate',
               'wine glass',
               'cup',
               'fork',
               'knife',
               'spoon',
               'bowl',
               'banana',
               'apple',
               'sandwich',
               'orange',
               'broccoli',
               'carrot',
               'hot dog',
               'pizza',
               'donut',
               'cake',
               'chair',
               'couch',
               'potted plant',
               'bed',
               'mirror',
               'dining table',
               'window',
               'desk',
               'toilet',
               'door',
               'tv',
               'laptop',
               'mouse',
               'remote',
               'keyboard',
               'cell phone',
               'microwave',
               'oven',
               'toaster',
               'sink',
               'refrigerator',
               'blender',
               'book',
               'clock',
               'vase',
               'scissors',
               'teddy bear',
               'hair drier',
               'toothbrush')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]] * 5

    def __init__(self, split, **kwargs):
        super(COCODataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
