# [WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation]()

**Table of Contents**

- [WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation](#weaktr-exploring-plain-vision-transformer-for-weakly-supervised-semantic-segmentation)
  - [Abstract](#abstract)
  - [Setup](#setup)
    - [Environment](#environment)
    - [Data preparation](#data-preparation)
  - [Training](#training)
    - [Phase 1: End-to-End CAM Generation](#phase-1-end-to-end-cam-generation)
    - [Phase 2: Online Retraining](#phase-2-online-retraining)
  - [Evaluation](#evaluation)
  - [Main results](#main-results)
    - [Pascal VOC 2012 Dataset](#pascal-voc-2012-dataset)


## Abstract 

<img src="WeakTr.png" style="zoom: 50%;" />

This paper explores the properties of the plain Vision Transformer (ViT) for Weakly-supervised Semantic Segmentation (WSSS). The class activation map (CAM) is of critical importance for understanding a classification network and launching WSSS. We observe that different attention heads of ViT focus on different image areas. Thus a novel weight-based method is proposed to end-to-end estimate the importance of attention heads, while the self-attention maps are adaptively fused for high-quality CAM results that tend to have more complete objects. Besides, we propose a ViT-based gradient clipping decoder for online retraining with the CAM results to complete the WSSS task. We name this plain **Tr**ansformer-based **Weakly**-supervised learning framework WeakTr. It achieves the state-of-the-art WSSS performance on standard benchmarks, i.e., 78.4% mIoU on the val set of PASCAL VOC 2012 and 50.3% mIoU on the val set of COCO 2014.

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

### Data preparation

- Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012), it is suggested to make a soft link toward downloaded dataset.

```bash
ln -s $your_dataset_path/pcontext/VOCdevkit/VOC2012 data/voc12
```

## Training

### Phase 1: End-to-End CAM Generation

```bash
# Training
python main.py --model deit_small_WeakTr_patch16_224 \
                --batch-size 64 \
                --data-set VOC12 \
                --img-list voc12 \
                --img-ms-list data/voc12/ImageSets/Segmentation/train.txt \
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
                --img-ms-list data/voc12/ImageSets/Segmentation/train.txt \
                --output_dir $your_model_dir \
                --resume $Weight \
                --gen_attention_maps \
                --attention-type fused \
                --visualize-cls-attn \
                --patch-attn-refine \
                --cam-npy-dir $your_CAM_npy_dir \
                
# CRF post-processing
python evaluation.py --list data/voc12/ImageSets/Segmentation/train.txt \
                     --gt_dir data/voc12/SegmentationClassAug \
                     --img_dir data/voc12/JPEGImages \
                     --type npy \
                     --t 42 \
                     --predict_dir $your_CAM_npy_dir \
                     --out-crf \
                     --out-dir $your_CAM_label_dir \
```

We provide CAM label [here](https://drive.google.com/drive/folders/186J1QUITCYZ4jZDJYzs6sKvXbsgLNYHX?usp=share_link), the mIoU is **69%** in the trainset.

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
DATASET=$your_dataset_path PYTHONPATH=. WORK=$your_project_path python segm/eval/miou.py --window-batch-size 1 --multiscale \
$your_checkpoint_path \
--predict-dir $your_pred_npy_dir \
pascal_context
```

2. CRF post-processing

```bash
python -m segm.eval.make_crf \
--list ../data/voc12/ImageSets/Segmentation/val.txt \
--predict-dir $your_pred_npy_dir \
--predict-png-dir $your_pred_png_dir \
--img-path ../data/voc12/JPEGImages \
--gt-folder ../data/voc12/SegmentationClassAug \
```

3. Evaluation

```bash
python -m segm.eval.make_crf \
--list ../data/voc12/ImageSets/Segmentation/val.txt \
--predict-dir $your_pred_crf_dir \
--type png \
--img-path ../data/voc12/JPEGImages \
--gt-folder ../data/voc12/SegmentationClassAug \
```

## Main results

### Pascal VOC 2012 Dataset

|                           Method                           |                                             Checkpoint                                             | Val mIoU | Pseudo-mask | Train mIoU |
|:----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:| :---------: | :--------: | :--------: |
|                           WeakTr                           | [Google Drive](https://drive.google.com/file/d/11n5gKLVeq7yXgya17OodyKeAciUfc_Ax/view?usp=sharing) | 74.0% | [Google Drive](https://drive.google.com/drive/folders/16QcrPxc2DabCUEUqOiPPs38oI7MWbrK2?usp=share_link) | 76.3% |
| WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/11n5gKLVeq7yXgya17OodyKeAciUfc_Ax/view?usp=sharing) | **78.4%** | [Google Drive]() | **80.3%** |

