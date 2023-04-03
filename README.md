# [WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation]()



**Table of Contents**

- [WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation](#weaktr-exploring-plain-vision-transformer-for-weakly-supervised-semantic-segmentation)
  - [Highlight](#highlight)
  - [Abstract](#abstract)
  - [Setup](#setup)
    - [Environment](#environment)
    - [Data preparation](#data-preparation)
  - [Training](#training)
    - [Phase 1: End-to-End CAM Generation](#phase-1-end-to-end-cam-generation)
    - [Phase 2: Online Retraining](#phase-2-online-retraining)
  - [Evaluation](#evaluation)
  - [Main results](#main-results)
    - [CAM Generation](#cam-generation)
    - [Online Retraining](#online-retraining)
  - [Citation](#citation)

## Highlight



<div align=center><img src="img/miou_compare.png" width="500px"></div>

- The proposed WeakTr fully explores the potential of plain ViT in the WSSS domain. State-of-the-art results are achieved on both challenging WSSS benchmarks, with **74.0%** mIoU on PASCAL VOC 2012 and **46.9%** on COCO 2014 validation sets respectively, significantly surpassing previous methods.
- The proposed WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) based on the improved ViT pretrained on ImageNet-21k and fine-tuned on ImageNet-1k performs better with **78.4%** mIoU on PASCAL VOC 2012 and **50.3%** on COCO 2014 validation sets respectively.

## Abstract 


This paper explores the properties of the plain Vision Transformer (ViT) for Weakly-supervised Semantic Segmentation (WSSS). The class activation map (CAM) is of critical importance for understanding a classification network and launching WSSS. We observe that different attention heads of ViT focus on different image areas. Thus a novel weight-based method is proposed to end-to-end estimate the importance of attention heads, while the self-attention maps are adaptively fused for high-quality CAM results that tend to have more complete objects. Besides, we propose a ViT-based gradient clipping decoder for online retraining with the CAM results to complete the WSSS task. We name this plain **Tr**ansformer-based **Weakly**-supervised learning framework WeakTr. It achieves the state-of-the-art WSSS performance on standard benchmarks, i.e., 78.4% mIoU on the val set of PASCAL VOC 2012 and 50.3% mIoU on the val set of COCO 2014.

**Step1: End-to-End CAM Generation**

<div align=center><img src="img/WeakTr.png" width="800px"></div>

**Step2: Online Retraining with Gradient Clipping Decoder**

<div align=center><img src="img/clip_grad_decoder.png" width="800px"></div>


## Setup

### Environment

```bash
conda create --name weaktr python=3.7
conda activate weaktr

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
pip install -r requirements.txt
```

Then, install [mmcv==1.4.0](https://github.com/open-mmlab/mmcv) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) following the official instruction.

```bash
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmsegmentation
```
And install `pydensecrf` from source.

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
### Data preparation

**Pascal VOC 2012**
- First download the Pascal VOC 2012 datasets use the scripts in the `data` dir.

```bash
cd data
sh download_and_convert_voc12.sh
```
- Then download SBD annotations from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip).


The folder structure is assumed to be:
```bash
- data
  - download_and_convert_voc12.sh
  + voc12
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + SegmentationClassAug
- voc12
  - cls_labels.npy
  - train_aug_id.txt
  - train_id.txt
  - val_id.txt
```

**COCO 2014**
- First download the COCO 2014 datasets use the scripts in the `data` dir.

```bash
cd data
sh download_and_convert_coco.sh
```
The folder structure is assumed to be:
```bash
- data
  - download_and_convert_coco.sh
  - voc12
  + coco
    + images
    + voc_format
      + class_labels
      + train.txt
      + val.txt
- coco
  - cls_labels.npy
  - train_id.txt
  - val_id.txt
```

## Training

### Phase 1: End-to-End CAM Generation

```bash
# Training
python main.py --model deit_small_WeakTr_patch16_224 \
                --batch-size 64 \
                --data-set VOC12 \
                --img-list voc12 \
                --img-ms-list voc12/train_id.txt \
                --gt-dir SegmentationClass \
                --scales 1.0 \
                --cam-npy-dir $your_cam_dir \
                --visualize-cls-attn \
                --patch-attn-refine \
                --data-path data/voc12 \
                --output_dir $your_output_dir \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                --if_eval_miou \
                --lr 4e-4 \
                --seed 504 \
                --extra-token

# Generate CAM
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 0.8 1.2 \
                --img-list voc12 \
                --data-path data/voc12 \
                --img-ms-list voc12/trian_aug_id.txt \
                --gt-dir SegmentationClassAug \
                --output_dir $your_model_dir \
                --resume $your_checkpoint_path \
                --gen_attention_maps \
                --attention-type fused \
                --visualize-cls-attn \
                --patch-attn-refine \
                --cam-npy-dir $your_CAM_npy_dir \
                
# CRF post-processing
python evaluation.py --list voc12/train_aug_id.txt \
                     --data-path data/voc12 \
                     --gt_dir SegmentationClassAug \
                     --img_dir JPEGImages \
                     --type npy \
                     --t 42 \
                     --predict_dir $your_CAM_npy_dir \
                     --out-crf \
                     --out-dir $your_CAM_label_dir \
```

We store the [best checkpoint](https://drive.google.com/file/d/1TW6HSSOnhzdAHpUrarM8-nhJTV3MvIgM/view?usp=share_link) of CAM generation and the [CAM label](https://drive.google.com/drive/folders/186J1QUITCYZ4jZDJYzs6sKvXbsgLNYHX?usp=share_link) for Online Retraining in Google Drive , the mIoU of the CAM label is **69%** in the trainset.

### Phase 2: Online Retraining

```bash
cd OnlineRetraining

MASTER_PORT=10201 DATASET=$your_dataset_path WORK=$your_project_path python -m segm.train \
--log-dir $your_log_dir \
--dataset pascal_context --backbone $your_model_name --decoder mask_transformer \
--batch-size 4 --epochs 100 -lr 1e-4 \
--num-workers 2 --eval-freq 1 \
--ann-dir $your_CAM_label_dir \
--start-value 1.2 --patch-size 120 \	

```

## Evaluation

```bash
cd OnlineRetraining
```

1. Multi-scale Evaluation 

```bash
MASTER_PORT=10201 DATASET=$your_dataset_path PYTHONPATH=. WORK=$your_project_path python segm/eval/miou.py --window-batch-size 1 --multiscale \
$your_checkpoint_path \
--predict-dir $your_pred_npy_dir \
pascal_context
```

2. CRF post-processing

```bash
python -m segm.eval.make_crf \
--list ../voc12/val_id.txt \
--data-path ../data/voc12 \
--predict-dir $your_pred_npy_dir \
--predict-png-dir $your_pred_png_dir \
--img-path JPEGImages \
--gt-folder SegmentationClassAug \
```

3. Evaluation

```bash
python -m segm.eval.make_crf \
--list ../voc12/val_id.txt \
--data-path ../data/voc12 \
--predict-dir $your_pred_crf_dir \
--type png \
--img-path JPEGImages \
--gt-folder SegmentationClassAug \
```

## Main results

### CAM Generation

|     Dataset     |                          Checkpoint                          |                          CAM_Label                           | Train mIoU |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| Pascal VOC 2012 | [Google Drive](https://drive.google.com/file/d/1TW6HSSOnhzdAHpUrarM8-nhJTV3MvIgM/view?usp=share_link) | [Google Drive](https://drive.google.com/drive/folders/186J1QUITCYZ4jZDJYzs6sKvXbsgLNYHX?usp=share_link) |   69.0%    |
|    COCO 2014    | [Google Drive](https://drive.google.com/file/d/1tFUDIQDXuD1f8MhWAza-jJP1hYeYhvSb/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/16_fRt5XfgzueEcmoRSFAHiI3rKUYz20r/view?usp=share_link) |   41.9%    |

### Online Retraining

|                        Dataset                        |                        Method                           |                                             Checkpoint                                             | Val mIoU | Pseudo-mask | Train mIoU |
|:----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:| :---------: | :--------: | :--------: |------------------------------------------------------------|
|                           Pascal VOC 2012                           |                           WeakTr                           | [Google Drive](https://drive.google.com/file/d/11n5gKLVeq7yXgya17OodyKeAciUfc_Ax/view?usp=share_link) | 74.0% | [Google Drive](https://drive.google.com/drive/folders/1I9GtlhcRDCA2i6C0s1I_OUx5S0H6ruL0?usp=share_link) | 76.3% |
| Pascal VOC 2012 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/1m8bBxjrqstVwwUi2fAiqtRmIk6COTxGy/view?usp=share_link) | **78.4%** | [Google Drive](https://drive.google.com/drive/folders/1MZAWfTB-PQGLcOYHHREna8bBIlSAKzn0?usp=share_link) | **80.3%** |
| COCO 2014 | WeakTr | [Google Drive](https://drive.google.com/file/d/1iZN0Gcg_uVRUgxlmQs7suAGjv6bsj4EX/view?usp=share_link) | 46.9% | [Google Drive](https://drive.google.com/file/d/1qZaiKqqAWFfY_-yxqcv8T29o8z9KytbJ/view?usp=share_link) | 48.9% |
| COCO 2014 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/13Gr_S8pG42IWDwcvvLcPmSfCW_MSa8fL/view?usp=share_link) | **50.3%** | [Google Drive](https://drive.google.com/file/d/1p-t-4pPIpJZDmj-oJ16PKVG3Yu8sNiaF/view?usp=share_link) | **51.3%** |

## Citation