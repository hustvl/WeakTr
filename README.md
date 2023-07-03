<div align="center">
<h1>WeakTr </h1>
<h3>Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation</h3>

[Lianghui Zhu](https://github.com/Unrealluver)<sup>1</sup> \*, [Yingyue Li](https://github.com/Yingyue-L)<sup>1</sup> \*, [Jiemin Fang](https://jaminfong.cn)<sup>1</sup>, [Xinggang Wang](https://scholar.google.com/citations?user=qNCTLV0AAAAJ&hl=zh-CN)<sup>1 :email:</sup>, Yan Liu<sup>2</sup>, Hao Xin<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>
 
<sup>1</sup> School of EIC, Huazhong University of Science & Technology, <sup>2</sup> Ant Group 

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ArXiv Preprint ([arXiv 2304.01184](https://arxiv.org/abs/2304.01184))


</div>


## Highlight



<div align=center><img src="img/miou_compare.png" width="500px"></div>

- The proposed WeakTr fully explores the potential of plain ViT in the WSSS domain. State-of-the-art results are achieved on both challenging WSSS benchmarks, with **74.0%** mIoU on PASCAL VOC 2012 and **46.9%** on COCO 2014 validation sets respectively, significantly surpassing previous methods.
- The proposed WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) based on the improved ViT pretrained on ImageNet-21k and fine-tuned on ImageNet-1k performs better with **78.4%** mIoU on PASCAL VOC 2012 and **50.3%** on COCO 2014 validation sets respectively.

## Introduction 

This paper explores the properties of the plain Vision Transformer (ViT) for Weakly-supervised Semantic Segmentation (WSSS). The class activation map (CAM) is of critical importance for understanding a classification network and launching WSSS. We observe that different attention heads of ViT focus on different image areas. Thus a novel weight-based method is proposed to end-to-end estimate the importance of attention heads, while the self-attention maps are adaptively fused for high-quality CAM results that tend to have more complete objects. 

**Step1: End-to-End CAM Generation**

<div align=center><img src="img/WeakTr.png" width="800px"></div>


Besides, we propose a ViT-based gradient clipping decoder for online retraining with the CAM results to complete the WSSS task. We name this plain **Tr**ansformer-based **Weak**ly-supervised learning framework WeakTr. It achieves the state-of-the-art WSSS performance on standard benchmarks, i.e., 78.4% mIoU on the val set of PASCAL VOC 2012 and 50.3% mIoU on the val set of COCO 2014.

**Step2: Online Retraining with Gradient Clipping Decoder**

<div align=center><img src="img/clip_grad_decoder.png" width="800px"></div> 





## Getting Started
- [Installation](docs/Install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Training](docs/Training.md)
- [Evaluate](docs/Evaluate.md)


## Main results

### Step1: End-to-End CAM Generation

|     Dataset     |     Method     |                          Checkpoint                          |                          CAM_Label                           | Train mIoU |
| :-------------: | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| Pascal VOC 2012 | WeakTr | [Google Drive](https://drive.google.com/file/d/19XEmgQKuTZQ2YQTncgCnttvXeBPhW-B-/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/12fO9K7uBFT08lUQb6rCj9wp_Mf8efa7F/view?usp=drive_link) |   69.3%    |
| COCO 2014 |    WeakTr    | [Google Drive](https://drive.google.com/file/d/1tFUDIQDXuD1f8MhWAza-jJP1hYeYhvSb/view?usp=share_link) | [Google Drive](https://drive.google.com/file/d/16_fRt5XfgzueEcmoRSFAHiI3rKUYz20r/view?usp=share_link) |   41.9%    |

### Step2: Online Retraining with Gradient Clipping Decoder

|                        Dataset                        |                        Method                           |                                             Checkpoint                                             | Val mIoU | Pseudo-mask | Train mIoU |
|:----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:| :---------: | :--------: | :--------: |------------|
|                           Pascal VOC 2012                           |                           WeakTr                          | [Google Drive](https://drive.google.com/file/d/1ONY86MyHSjjoKGdPI_0wmxQ_Go-39W2C/view?usp=share_link) | 74.0% | [Google Drive](https://drive.google.com/file/d/1Z6ioXhy6L4_2XMrj2dk7QiYPzZnWdbt-/view?usp=share_link) | 76.5%      |
| Pascal VOC 2012 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/1lLY4iNplYTDTTOtnE10OzdjtMDZZfJX0/view?usp=sharing) | **78.4%** | [Google Drive](https://drive.google.com/file/d/1kEhlTVZvIqMGD5ck8UDTxWxXD7UMf3VZ/view?usp=share_link) | **80.3%**  |
| COCO 2014 | WeakTr | [Google Drive](https://drive.google.com/file/d/1iZN0Gcg_uVRUgxlmQs7suAGjv6bsj4EX/view?usp=share_link) | 46.9% | [Google Drive](https://drive.google.com/file/d/1qZaiKqqAWFfY_-yxqcv8T29o8z9KytbJ/view?usp=share_link) | 48.9%      |
| COCO 2014 | WeakTr![](http://latex.codecogs.com/svg.latex?^{\dagger}) | [Google Drive](https://drive.google.com/file/d/13Gr_S8pG42IWDwcvvLcPmSfCW_MSa8fL/view?usp=share_link) | **50.3%** | [Google Drive](https://drive.google.com/file/d/1p-t-4pPIpJZDmj-oJ16PKVG3Yu8sNiaF/view?usp=share_link) | **51.3%**  |

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
```
@article{zhu2023weaktr,
      title={WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation}, 
      author={Lianghui Zhu and Yingyue Li and Jiemin Fang and Yan Liu and Hao Xin and Wenyu Liu and Xinggang Wang},
      year={2023},
      journal={arxiv:2304.01184},
}
```

