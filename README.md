# [WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation](https://arxiv.org/abs/2304.01184)



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
pip install mmsegmentation==0.30.0
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
- Then download SBD annotations from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip) and download the Pascal VOC 2012 [CAM_Label](https://drive.google.com/file/d/1iDI7NPO0qrTz4dsnCtyGlPZxrAgp2U77/view?usp=share_link).

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
        + WeakTr_CAMlb_wCRF
- voc12
  - cls_labels.npy
  - train_aug_id.txt
  - train_id.txt
  - val_id.txt
```

**COCO 2014**
- First download the COCO 2014 datasets use the scripts in the `data` dir and download the COCO 2014 [CAM_Label](https://drive.google.com/file/d/16_fRt5XfgzueEcmoRSFAHiI3rKUYz20r/view?usp=share_link).

```bash
cd data
sh download_and_convert_coco.sh
cp ../coco/val_5000.txt coco/voc_format
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
      + val_5000.txt
      + WeakTr_CAMlb_wCRF_COCO
- coco
  - cls_labels.npy
  - train_id.txt
  - train_1250_id.txt
  - val_id.txt
  - val_5000.txt
```

## Training

### Phase 1: End-to-End CAM Generation

#### Pascal VOC 2012
```bash
# Training
python main.py  --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12 \
                --img-ms-list voc12/train_id.txt \
                --cam-npy-dir WeakTr_results/WeakTr/attn-patchrefine-npy \
                --output_dir WeakTr_results/WeakTr \
                --lr 4e-4 \

# Generate CAM
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/train_aug_id.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir WeakTr_results/WeakTr/attn-patchrefine-npy-ms \
                --output_dir WeakTr_results/WeakTr \
                --resume WeakTr_results/WeakTr/checkpoint_best_mIoU.pth
                
# CRF post-processing
python evaluation.py --list voc12/train_aug_id.txt \
                     --data-path data \
                     --type npy \
                     --predict_dir WeakTr_results/WeakTr/attn-patchrefine-npy-ms \
                     --out-dir WeakTr_results/WeakTr/pseudo-mask-ms-crf \
                     --t 42 \
                     --out-crf

```
#### COCO 2014
```bash
# Training
python main.py  --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set COCO \
                --img-ms-list coco/train_1250_id.txt \
                --gt-dir voc_format/class_labels \
                --cam-npy-dir WeakTr_results_coco/WeakTr/attn-patchrefine-npy \
                --output_dir WeakTr_results_coco/WeakTr \
                --lr 5e-4 \
                
# Generate CAM
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set COCOMS \
                --img-ms-list coco/train_id.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir WeakTr_results_coco/WeakTr/attn-patchrefine-npy-ms \
                --output_dir WeakTr_results_coco/WeakTr \
                --resume WeakTr_results_coco/WeakTr/checkpoint_best_mIoU.pth
                
# CRF post-processing
python evaluation.py --list coco/train_id.txt \
                     --data-path data \
                     --type npy \
                     --predict_dir WeakTr_results_coco/WeakTr/attn-patchrefine-npy-ms \
                     --out-dir WeakTr_results_coco/WeakTr/pseudo-mask-ms-crf \
                     --t 42 \
                     --out-crf
```
### Phase 2: Online Retraining
#### Pascal VOC 2012

```bash
cd OnlineRetraining

bash segm/dist_train.sh 1 \
--log-dir start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr \
--dataset pascal_voc --backbone deit_small_patch16_224 --decoder mask_transformer \
--ann-dir WeakTr_CAMlb_wCRF \
--start-value 1.2 --patch-size 120 \		
```

#### COCO 2014
```bash
cd OnlineRetraining

bash segm/dist_train.sh 4 \
--log-dir start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr \
--dataset coco --backbone deit_small_patch16_224 --decoder mask_transformer \
--ann-dir voc_format/WeakTr_CAMlb_wCRF_COCO \
--start-value 1.2 --patch-size 120 \	
```
## Evaluation
### Pascal VOC 2012

```bash
cd OnlineRetraining
```

1. Multi-scale Evaluation 

```bash
bash segm/dist_test.sh 4 \
--multiscale \
--predict-dir start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr/seg_prob_ms \
start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr/checkpoint.pth \
pascal_voc
```

2. CRF post-processing

```bash
python -m segm.eval.make_crf \
--list val.txt \
--data-path ../data \
--predict-dir start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr/seg_prob_ms \
--predict-png-dir start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr/seg_pred_ms \
--num-cls 21 \
--dataset voc12
```

### COCO 2014

```bash
cd OnlineRetraining
```

1. Multi-scale Evaluation 

```bash
bash segm/dist_test.sh 4 \
--multiscale \
--predict-dir start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr/seg_prob_ms \
start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr/checkpoint.pth \
coco
```

2. CRF post-processing

```bash
python -m segm.eval.make_crf \
--list val.txt \
--data-path ../data \
--predict-dir start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr/seg_prob_ms \
--predict-png-dir start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr/seg_pred_ms \
--num-cls 91 \
--dataset coco
```
## Main results

### CAM Generation

|     Dataset     |                          Checkpoint                          |                          CAM_Label                           | Train mIoU |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| Pascal VOC 2012 | [Google Drive](https://drive.google.com/file/d/1TW6HSSOnhzdAHpUrarM8-nhJTV3MvIgM/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/1iDI7NPO0qrTz4dsnCtyGlPZxrAgp2U77/view?usp=share_link) |   69.0%    |
|    COCO 2014    | [Google Drive](https://drive.google.com/file/d/1tFUDIQDXuD1f8MhWAza-jJP1hYeYhvSb/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/16_fRt5XfgzueEcmoRSFAHiI3rKUYz20r/view?usp=share_link) |   41.9%    |

### Online Retraining

|                        Dataset                        |                        Method                           |                                             Checkpoint                                             | Val mIoU | Pseudo-mask | Train mIoU |
|:----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:| :---------: | :--------: | :--------: |------------|
|                           Pascal VOC 2012                           |                           WeakTr                           | [Google Drive](https://drive.google.com/file/d/11n5gKLVeq7yXgya17OodyKeAciUfc_Ax/view?usp=share_link) | 74.0% | [Google Drive](https://drive.google.com/file/d/1Z6ioXhy6L4_2XMrj2dk7QiYPzZnWdbt-/view?usp=share_link) | 76.5%      |
| Pascal VOC 2012 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/1m8bBxjrqstVwwUi2fAiqtRmIk6COTxGy/view?usp=share_link) | **78.4%** | [Google Drive](https://drive.google.com/file/d/1kEhlTVZvIqMGD5ck8UDTxWxXD7UMf3VZ/view?usp=share_link) | **80.3%**  |
| COCO 2014 | WeakTr | [Google Drive](https://drive.google.com/file/d/1iZN0Gcg_uVRUgxlmQs7suAGjv6bsj4EX/view?usp=share_link) | 46.9% | [Google Drive](https://drive.google.com/file/d/1qZaiKqqAWFfY_-yxqcv8T29o8z9KytbJ/view?usp=share_link) | 48.9%      |
| COCO 2014 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/13Gr_S8pG42IWDwcvvLcPmSfCW_MSa8fL/view?usp=share_link) | **50.3%** | [Google Drive](https://drive.google.com/file/d/1p-t-4pPIpJZDmj-oJ16PKVG3Yu8sNiaF/view?usp=share_link) | **51.3%**  |

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
```
@article{zhu2023weaktr,
      title={WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation}, 
      author={Lianghui Zhu and Yingyue Li and Jieming Fang and Yan Liu and Hao Xin and Wenyu Liu and Xinggang Wang},
      year={2023},
      journal={arxiv:2304.01184},
}
```

