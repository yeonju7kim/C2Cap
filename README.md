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
git clone https://github.com/davidnvq/grit.git
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
* Install Deformable Attention:
```shell
cd models/ops/
python setup.py build develop
python test.py
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

The model is trained with default settings in the configurations file in `configs/caption/coco_config.yaml`:
The training process takes around 16 hours on a machine with 8 A100 GPU.
We also provide the code for extracting pretrained features (freezed object detector), which will speed up the training significantly.

* With default configurations (e.g., 'parallel Attention', pretrained detectors on VG or 4DS, etc):
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path

# with pretrained object detector on Visual Genome
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=vg_detector_path
```

<!-- * **More configurations will be added here for obtaining ablation results**. -->
* To freeze the backbone and detector, we can extract the region features and initial grid features first, saving it to `dataset.hdf5_path` in the config file.
Then we can run the following script to train the model:
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path \
optimizer.freezing_xe_epochs=10 \
optimizer.freezing_sc_epochs=10 \
optimizer.finetune_xe_epochs=0 \
optimizer.finetune_sc_epochs=0 \
optimizer.freeze_backbone=True \
optimizer.freeze_detector=True
```

### Evaluation

The evaluation will be run on a single GPU.
* Evaluation on **Karapthy splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```
* Evaluation on the **online splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption_online.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption_online.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```

### Inference on RGB Image

* Perform Inference for a single image using the script `inference_caption.py`:
```
python inference_caption.py +img_path='notebooks/COCO_val2014_000000000772.jpg' \
+vocab_path='path_to_annotations/vocab.json' \
exp.checkpoint='path_to_caption_checkpoint'
```
*  Perform Inference for a single image using the Jupyter notebook `notebooks/Inference.ipynb`
```shell
# Require installing Jupyter(lab)
pip install jupyterlab

cd notebooks
# Open jupyter notebook
jupyter lab
```


## Citation
If you find this code useful, please kindly cite the paper with the following bibtex:
```bibtex
@article{nguyen2022grit,
  title={GRIT: Faster and Better Image captioning Transformer Using Dual Visual Features},
  author={Nguyen, Van-Quang and Suganuma, Masanori and Okatani, Takayuki},
  journal={arXiv preprint arXiv:2207.09666},
  year={2022}
}
```

## Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.