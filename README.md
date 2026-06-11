# ICASSP2026-ALIGNCLIP
Official implementation of the paper[^2]: 
**ALIGNCLIP: MINING AND ALIGNING MULTI-SCALE VISION-LANGUAGE FEATURES FOR ZERO-SHOT SEMANTIC SEGMENTATION**

## Environment
+ Install pytorch 
`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch`
+ Install the mmsegmentation library and other dependencies
`pip install mmcv-full==1.4.4 mmsegmentation==0.24.0 pip install scipy timm==0.3.2`
>In principle, `mmcv-full` can be installed in versions higher than 1.4.4. If you encounter problems with environment configuration, we recommend that you consult large language models for assistance.

<br><br>

---

## Pretrained Weights

[Pretrained Weights Download](https://drive.google.com/drive/folders/1G-_zOtPM27FXN4bwWXF1nly8ut1G5u4s?usp=drive_link)

Reference results:

| Setting | Method | Dataset | mIoU(S) | mIoU(U) | hIoU |
| --- | --- | --- | ---: | ---: | ---: |
| inductive | AlignCLIP (ours) | PASCAL VOC 2012 | 93.1 | 84.5 | 88.6 |
| inductive | AlignCLIP (ours) | COCO-Stuff 164K | 41.5 | 44.2 | 42.8 |
| inductive | AlignCLIP (ours) | PASCAL Context[^1] | 55.1 | 50.0 | 52.4 |
| transductive | AlignCLIP (ours) | PASCAL VOC 2012 | 94.0 | 94.0 | 94.0 |
| transductive | AlignCLIP (ours) | COCO-Stuff 164K | 42.4 | 60.9 | 50.0 |
| transductive | AlignCLIP (ours) | PASCAL Context | 56.0 | 54.2 | 55.0 |

[^1]: Results on PASCAL Context were not reported in the original paper and are provided only as a follow-up baseline for comparison and reference.

## Prepare the dataset
Please according to the mmsegmentation library's official document operation:
[MMSegmentation Dataset Preparation Guide](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md)

<br><br>

---

## Prepare for the pre-training of CLIP weights
+ Download link: [CLIP ViT-B/16 Pretrained Weights](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)
+ Change the loading directory. For example, when training on the COCO dataset, you need to change the path `pretrained =` under `configs\coco\vpt_seg_zero_vit -b_512x512_80k_12_100_multi-.py` to `path/your_pretrained_model`

<br><br>

---


## training (inductive)
+ Training is conducted using multiple GPU. The following takes 4x3090 GPUs as an example:
The first parameter is the directory of the configuration file, the second parameter indicates the number of GPUs, and the third parameter represents the save path
```
bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py 4 Path/to/coco/zero_12_100
bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py 4 Path/to/voc12/zero_12_10
```
+ Training with a single GPU:
There is no need to display the specified number of GPU
```
bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100
bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py Path/to/voc12/zero_12_10
```
You can also directly use the following command:
```
python train.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100
python train.py configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py Path/to/voc12/zero_12_10
```

## training (transductive)
```
bash dist_train.sh ./configs/coco/vpt_seg_zero_vit-b_512x512_40k_12_100_multi_st.py Path/to/coco/zero_12_100_st --load-from=Path/to/coco/zero_12_100/iter_40000.pth
bash dist_train.sh ./configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_st.py Path/to/voc12/zero_12_10_st --load-from=Path/to/voc12/zero_12_10/iter_10000.pth
```

<br><br>

---

## Citation
If you find this project useful, please consider citing:
```python
@INPROCEEDINGS{11460884,
  author={Mei, Lian and Yang, Fan and Chen, Yu and Wang, Ke and Huang, Enhao and Deng, Yuhui},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={AlignCLIP: Mining and Aligning Multi-Scale Vision-Language Features For Zero-Shot Semantic Segmentation}, 
  year={2026},
  volume={},
  number={},
  pages={5931-5935},
  keywords={Feeds;Antennas;Filtering;Circuits and systems;Filters;Pixel;Product development;Protocols;HTTP;Digital images;Zero-Shot Semantic Segmentation;Multi-Scale Feature;Semantic Alignment;Vision-Language Model},
  doi={10.1109/ICASSP55912.2026.11460884}}
```

<br><br>

---

## Acknowledgement
Our work is inspired by these assets. Please consider to cite them as well in your paper.
+ OpenAI's [CLIP](https://github.com/openai/CLIP)
```python
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={International Conference on Machine Learning},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:231591445}
}
```
+ CascadeCLIP: 
```python
@inproceedings{licascade,
  title={Cascade-CLIP: Cascaded Vision-Language Embeddings Alignment for Zero-Shot Semantic Segmentation},
  author={Li, Yunheng and Li, Zhong-Yu and Zeng, Quan-Sheng and Hou, Qibin and Cheng, Ming-Ming},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
+ ZegCLIP:
```python
@article{zhou2022zegclip,
  title={ZegCLIP: Towards adapting CLIP for zero-shot semantic segmentation},
  author={Zhou, Ziqin and Lei, Yinjie and Zhang, Bowen and Liu, Lingqiao and Liu, Yifan},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
+ CLIP-RC:
```python
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Yi and Guo, Meng-Hao and Wang, Miao and Hu, Shi-Min},
    title     = {Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {3270-3280}
}
```


[^2]: This GitHub repository will be permanently discontinued from any further updates and is retained as a historical archive.

