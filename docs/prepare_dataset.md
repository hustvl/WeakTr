# Data preparation

## Pascal VOC 2012
- First download the Pascal VOC 2012 datasets use the scripts in the `data` dir.

```bash
cd data
sh download_and_convert_voc12.sh
```
- Then download SBD annotations from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip) and download the Pascal VOC 2012 [CAM_Label](https://drive.google.com/file/d/1iDI7NPO0qrTz4dsnCtyGlPZxrAgp2U77/view?usp=share_link).

The folder structure is assumed to be:
```bash
WeakTr
├── data
│   ├── download_and_convert_voc12.sh
│   ├── voc12
│   │   ├── VOCdevkit
│   │   │   ├── VOC2012
│   │   │   │   ├── JPEGImages
│   │   │   │   ├── SegmentationClass
│   │   │   │   ├── SegmentationClassAug
│   │   │   │   ├── WeakTr_CAMlb_wCRF
├── voc12
│   ├── cls_labels.npy
│   ├── train_aug_id.txt
│   ├── train_id.txt
│   ├── val_id.txt
```

## COCO 2014 
- First download the COCO 2014 datasets use the scripts in the `data` dir and download the COCO 2014 [CAM_Label](https://drive.google.com/file/d/16_fRt5XfgzueEcmoRSFAHiI3rKUYz20r/view?usp=share_link).

```bash
cd data
sh download_and_convert_coco.sh
cp ../coco/val_5000.txt coco/voc_format
cp ../coco/val_id.txt coco/voc_format/val.txt
cp ../coco/train_id.txt coco/voc_format/train.txt
```
- Then download the COCO 2014 semantic segmentation labels from [here](https://drive.google.com/file/d/1JIvfoBwkxp2_DlEeszlXbe6L8qROjPSX/view?usp=drive_link)

The folder structure is assumed to be:
```bash
WeakTr
├── data
│   ├── download_and_convert_coco.sh
│   ├── voc12
│   ├── coco
│   │   ├── images
│   │   ├── voc_format
│   │   │   ├── class_labels
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── val_5000.txt
│   │   │   ├── WeakTr_CAMlb_wCRF_COCO
├── voc12
│   ├── cls_labels.npy
│   ├── train_aug_id.txt
│   ├── train_id.txt
│   ├── val_id.txt
├── coco
│   ├── cls_labels.npy
│   ├── train_id.txt
│   ├── train_1250_id.txt
│   ├── val_id.txt
│   ├── val_5000.txt
```