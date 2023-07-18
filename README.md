## Mitigating Dataset Bias in Image Captioning through CLIP Confounder-free Captioning Network

This is the code implementation for the paper titled: "Mitigating Dataset Bias in Image Captioning through CLIP Confounder-free Captioning Network" (Accepted to ICIP 2023) 

[//]: # ([[Arxiv]&#40;https://arxiv.org/abs/2207.09666&#41;].)


## Introduction

To solve the dataset bias problem, we approach from the causal inference perspective and design a causal graph. Based on the causal graph, we propose a novel method named C2Cap which is CLIP confounder-free captioning network. We use the global visual confounder to control the confounding factors in the image and train the model to produce debiased captions.


## Installation

### Requirements
* Python >= 3.9, CUDA >= 11.3
* PyTorch >= 1.12.0, torchvision >= 0.6.1
* Other packages: pycocotools, tensorboard, tqdm, h5py, nltk, einops, hydra, spacy, and timm

* First, clone the repository locally:
```shell
git clone https://github.com/yeonju7kim/C2Cap.git
cd grit
```
* Then, create an environment and install PyTorch and torchvision:
```shell
conda create -n grit python=3.9
conda activate grit
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# ^ if the CUDA version is not compatible with your system; visit pytorch.org for compatible matches.
```
* Install other requirements:
```shell
pip install -r requirements.txt
python -m spacy download en
```

## Usage

> Currently, the README and source code are under its initial version. The cleaned and detailed version may be updated soon.

### Data preparation

Download and extract COCO 2014 for image captioning including train, val, and test images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco_caption/
├── annotations/  # annotation json files and Karapthy files
├── train2014/    # train images
├── val2014/      # val images
└── test2014/     # test images
```
* Copy the files in `data/` to the above `annotations` folder. It includes `vocab.json` and some files containing Karapthy ids.

### Training

* Training on **Karapthy splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```

### Evaluation

* Evaluation on **Karapthy splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```

## Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.