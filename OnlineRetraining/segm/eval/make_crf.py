import joblib
import os
import numpy as np
import argparse
import torch
from PIL import Image
import pandas as pd
from segm.eval.densecrf import crf_inference_voc12, crf_inference_coco
from pathlib import Path

import torch.nn.functional as F

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
categories_coco = ['background',
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
                   'toothbrush']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train_id.txt', type=str)
    parser.add_argument("--data-path", default='data/voc12')
    parser.add_argument("--predict-dir", default=None, type=str)
    parser.add_argument("--predict-png-dir", default=None, type=str)
    parser.add_argument("--img-path", default=None, type=str)
    parser.add_argument("--n-jobs", default=10, type=int)
    parser.add_argument("--num-cls", default=21, type=int)
    parser.add_argument("--gt-folder", default='data/voc12/SegmentationClassAug', type=str)
    parser.add_argument("--dataset", default="voc12", type=str)
    parser.add_argument("--type", default="npy", type=str)

    args = parser.parse_args()
    if args.dataset == "voc12":
        args.data_path = Path(args.data_path) / "voc12" if "voc12" not in args.data_path else args.data_path
        args.data_path = Path(args.data_path) / "VOCdevkit" / "VOC2012"
        args.gt_folder = args.data_path / "SegmentationClassAug"
        args.img_path = args.data_path / "JPEGImages"
        args.list = os.path.join(args.data_path / "ImageSets" / "Segmentation", args.list)

    if args.dataset == "coco":
        args.data_path = Path(args.data_path) / "coco" if "coco" not in args.data_path else args.data_path
        args.gt_folder = args.data_path / "voc_format" / "class_labels"
        args.img_path = args.data_path / "images"
        args.list = os.path.join(args.data_path / "voc_format", args.list)

    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    if args.predict_png_dir is not None:
        Path(args.predict_png_dir).mkdir(parents=True, exist_ok=True)
    num_cls = args.num_cls


    def compute(i):
        if "npy" in args.type:
            path = os.path.join(args.predict_dir, name_list[i] + ".npy")
            seg_prob = np.load(path, allow_pickle=True).item()
            if args.type == "npypng":
                keys = seg_prob["keys"]
                predict = np.argmax(seg_prob["prob"], axis=0)
                predict = keys[predict].astype(np.uint8)
            else:
                orig_image = np.asarray(Image.open(os.path.join(args.img_path, name_list[i] + ".jpg")).convert("RGB"))
                keys = seg_prob["keys"]
                seg_prob = seg_prob["prob"]

                if args.dataset == "voc12":
                    seg_prob = crf_inference_voc12(orig_image, seg_prob, labels=seg_prob.shape[0])
                elif args.dataset == "coco":
                    # seg_prob = F.softmax(torch.tensor(seg_prob), dim=0).cpu().numpy()
                    seg_prob = crf_inference_coco(orig_image, seg_prob, labels=seg_prob.shape[0])
                else:
                    raise "dataset error"

                predict = np.argmax(seg_prob, axis=0)
                predict = keys[predict].astype(np.uint8)
        elif args.type == "png":
            path = os.path.join(args.predict_dir, name_list[i] + ".png")
            predict = np.array(Image.open(path))

        if args.predict_png_dir is not None:
            predict_img = Image.fromarray(predict.astype(np.uint8))
            predict_img.save(os.path.join(args.predict_png_dir, name_list[i] + ".png"))

        if "test" not in args.list:
            gt_file = os.path.join(args.gt_folder, name_list[i] + ".png")
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            p_list, t_list, tp_list = [0] * args.num_cls, [0] * args.num_cls, [0] * args.num_cls
            for i in range(args.num_cls):
                p_list[i] += np.sum((predict == i) * cal)
                t_list[i] += np.sum((gt == i) * cal)
                tp_list[i] += np.sum((gt == i) * mask)

            return p_list, t_list, tp_list

    if "test" in args.list:
        joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(compute)(i) for i in range(len(name_list))]
        )
        import sys
        sys.exit()
    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(compute)(i) for i in range(len(name_list))]
    )
    p_lists, t_lists, tp_lists = zip(*results)
    TP = [0] * num_cls
    P = [0] * num_cls
    T = [0] * num_cls
    for idx in range(len(name_list)):
        p_list = p_lists[idx]
        t_list = t_lists[idx]
        tp_list = tp_lists[idx]
        for i in range(num_cls):
            TP[i] += tp_list[i]
            P[i] += p_list[i]
            T[i] += t_list[i]

        IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    Prediction = []
    Recall = []
    for i in range(num_cls):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i]))
        T_TP.append(T[i] / (TP[i]))
        P_TP.append(P[i] / (TP[i]))
        FP_ALL.append((P[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        FN_ALL.append((T[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        Prediction.append(TP[i] / P[i])
        Recall.append(TP[i] / T[i])
    loglist = {}
    for i in range(num_cls):
        if num_cls == 21:
            loglist[categories[i]] = IoU[i] * 100
        else:
            loglist[categories_coco[i]] = IoU[i] * 100
    miou = np.nanmean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.nanmean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.nanmean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    prediction = np.nanmean(np.array(Prediction))
    loglist['Prediction'] = prediction * 100
    recall = np.nanmean(np.array(Recall))
    loglist['Recall'] = recall * 100
    for i in range(num_cls):
        if num_cls == 21:
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        else:
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100))
    print('\n======================================================')
    print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    print('\n')
    print(f'FP = {fp * 100}, FN = {fn * 100}')
    print(f'Prediction = {prediction * 100}, Recall = {recall * 100}')
