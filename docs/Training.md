# Training

## Phase 1: End-to-End CAM Generation

### Pascal VOC 2012
```bash
# Training
# WeakTrV1
CUDA_VISIBLE_DEVICES=0 python main.py  --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12 \
                --img-ms-list voc12/train_id.txt \
                --cam-npy-dir WeakTr_results/WeakTr/attn-patchrefine-npy \
                --output_dir WeakTr_results/WeakTr \
                --lr 4e-4 \

# WeakTrV2
CUDA_VISIBLE_DEVICES=2 python main.py  --model deit_small_WeakTr_AAF_AttnFeat_patch16_224 \
                --data-path data \
                --data-set VOC12 \
                --img-ms-list voc12/train_id.txt \
                --cam-npy-dir WeakTr_results/WeakTrV2/attn-patchrefine-npy \
                --output_dir WeakTr_results/WeakTrV2 \
                --reduction 8 \
                --pool-type max \
                --lr 6e-4 \
                --weight-decay 0.03 \
                --no-deterministic \
                --no-skiplist \
                --no-filter-bias-and-bn \

# Generate CAM
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 \
main.py --model deit_small_WeakTr_AAF_AttnFeat_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/train_aug_id.txt \
                --scales 1.0 1.2 \
                --gen_attention_maps \
                --cam-npy-dir WeakTr_results/WeakTrV2/attn-patchrefine-npy-ms \
                --output_dir WeakTr_results/WeakTrV2 \
                --resume WeakTr_results/WeakTrV2/checkpoint_best_mIoU.pth \
                --reduction 8 \
                --pool-type max \

python evaluation.py --list voc12/train_id.txt \
                     --data-path data \
                     --type npy \
                     --predict_dir WeakTr_results/WeakTrV2/attn-patchrefine-npy-ms \
                     --curve True \
                     --start 40 \

# CRF post-processing
python evaluation.py --list voc12/train_id.txt \
                     --data-path data \
                     --type npy \
                     --predict_dir WeakTr_results/WeakTrV2/attn-patchrefine-npy-ms \
                     --out-dir WeakTr_results/WeakTrV2/pseudo-mask-ms-crf \
                     --curve True \
                     --out-crf \
                     --start 40 \


python evaluation.py --list voc12/train_aug_id.txt \
                     --data-path data \
                     --type npy \
                     --predict_dir WeakTr_results/WeakTrV2/attn-patchrefine-npy-ms \
                     --out-dir WeakTr_results/WeakTrV2/pseudo-mask-ms-crf \
                     --t 41 \
                     --out-crf

```
### COCO 2014
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
## Phase 2: Online Retraining
### Pascal VOC 2012

```bash
cd OnlineRetraining

bash segm/dist_train.sh 1 \
--log-dir start1.2_patch120_seg_deit_small_patch16_224_voc_weaktr \
--dataset pascal_voc --backbone deit_small_patch16_224 --decoder mask_transformer \
--ann-dir WeakTr_CAMlb_wCRF \
--start-value 1.2 --patch-size 120 \		
```

### COCO 2014
```bash
cd OnlineRetraining

bash segm/dist_train.sh 4 \
--log-dir start1.2_patch120_seg_deit_small_patch16_224_COCO_weaktr \
--dataset coco --backbone deit_small_patch16_224 --decoder mask_transformer \
--ann-dir voc_format/WeakTr_CAMlb_wCRF_COCO \
--start-value 1.2 --patch-size 120 \	
```