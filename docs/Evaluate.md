# Evaluation
## Pascal VOC 2012

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

## COCO 2014

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