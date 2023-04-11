from pycocotools.coco import COCO
from pycocotools import mask
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

class COCOSegmentation(object):
    # these are the same as the PASCAL VOC dataset
    IGNORE_INDEX = 255
    def __init__(self, root_dir, split='train', year='2017', to_voc12=False):
        super(COCOSegmentation, self).__init__()
        ann_file = os.path.join(root_dir, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = os.path.join(root_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.ids = list(self.coco.imgs.keys())
        self.CAT_LIST = [0] + list(self.coco.getCatIds()) # add background 
        if to_voc12:
          self.CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] # COCO to pascalVOC12 categories mapping
        print(f'Categories: {self.CAT_LIST}')
    
    def generate_img_mask_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']

        # if img_metadata['height'] < 256 or img_metadata['width'] < 256:
        #     return None, None, None

        try:
            _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
            cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            _target = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width']))

            len_uniq_vals = len(np.unique(_target))
            # if len_uniq_vals < 2:
            #     return None, None, None

            return path, _img, _target
        except OSError:
            return None, None, None

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                c = self.IGNORE_INDEX
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

def create_pascal_label_colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def generate_pairs(coco, rgb_dir_name, mask_dir_name, save_dir_rgb, save_dir_mask):
    lines = []
    for i in tqdm(range(len(coco.ids))):
        fname, img, mask = coco.generate_img_mask_pair(i)
        if fname is None:
            continue
        fname = fname.split("_")[-1]
        img.save(os.path.join(save_dir_rgb, fname),format="jpeg")

        mask_fname = fname.split('.')[0] + '.png'
        mask.save(os.path.join(save_dir_mask, mask_fname),format="png")
        loc_pair = '{}\n'.format(fname.split('.')[0])
        # loc_pair = '{}/{} {}/{}\n'.format(rgb_dir_name, fname, mask_dir_name, mask_fname)
        lines.append(loc_pair)

    return lines


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--split", help="the desired split to convert.", type=str, default='train')
    parser.add_argument("--year", help="the desired coco dataset year.", type=str, default='2017')
    parser.add_argument("--to-voc12", help="wheater to convert to pascalvoc class set.", type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    root_dir = './coco/'
    assert args.year == '2014' or args.year == '2017', 'only coco14 and coco17 supported'

    # process the training split
    split = args.split
    year = args.year
    to_voc12 = args.to_voc12
    rgb_dir_name = 'images'
    mask_dir_name = 'voc_format/class_labels'
    txt_dir_name = 'voc_format'

    save_dir_rgb = root_dir + os.sep + rgb_dir_name
    save_dir_mask = root_dir + os.sep + mask_dir_name
    if not os.path.isdir(save_dir_rgb):
        os.makedirs(save_dir_rgb)

    if not os.path.isdir(save_dir_mask):
        os.makedirs(save_dir_mask)

    coco = COCOSegmentation(root_dir, split=split, year=year, to_voc12=to_voc12)

    lines = generate_pairs(coco, rgb_dir_name=rgb_dir_name, mask_dir_name=mask_dir_name,
                           save_dir_rgb=save_dir_rgb, save_dir_mask=save_dir_mask)

    with open(root_dir + os.sep + txt_dir_name + os.sep + '{}.txt'.format(split), 'w') as txt_file:
        for line in lines:
            txt_file.write(line)
