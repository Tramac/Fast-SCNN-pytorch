# Fast-SCNN: Fast Semantic Segmentation Network
A PyTorch implementation of [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/pdf/1902.04502) from the paper by Rudra PK Poudel, Stephan Liwicki.

<p align="center"><img width="100%" src="./png/Fast-SCNN.png" /></p>

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-fast-scnn'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#demo'>Demo</a>
- <a href='#results'>Results</a>
- <a href='#todo'>TO DO</a>
- <a href='#references'>Reference</a>

## Installation
- Python 3.x. Recommended using [Anaconda3](https://www.anaconda.com/distribution/)
- [PyTorch 1.0](https://pytorch.org/get-started/locally/). Install PyTorch by selecting your environment on the website and running the appropriate command. Such as:
  ```
  conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
  ```
- Clone this repository.
- Download the dataset by following the [instructions](#datasets) below.
- Note: For training, we currently support [cityscapes](https://www.cityscapes-dataset.com/), and aim to add [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/).

## Datasets
- You can download [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).
~~- To make things easy, we will provide bash scripts to handle the dataset downloads and setup for you. Come soon~:smile:~~

## Training-Fast-SCNN
- By default, we assume you have downloaded the cityscapes dataset in the `./datasets/citys` dir.
- To train Fast-SCNN using the train script the parameters listed in `train.py` as a flag or manually change them.
```Shell
python train.py --model fast_scnn --dataset citys
```

## Evaluation
To evaluate a trained network:
```Shell
python eval.py
```

## Demo
Running a demo:
```Shell
python demo.py --model fast_scnn --input-pic './png/berlin_000000_000019_leftImg8bit.png'
```

## Results
|Method|Dataset|crop_size|mIoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|
|Fast-SCNN(paper)|cityscapes||||
|Fast-SCNN(ours)|cityscapes||||

## TODO
- Still to come:
 ~~* [ ] Add dataset downloader~~
 * [ ] Train and eval
 * [ ] Support for the VOC, ADE20K dataset
 * [ ] Support [Visdom](https://github.com/facebookresearch/visdom)
 
## Authors
* [**Tramac**](https://github.com/Tramac)

## References
- Rudra PK Poudel. et al. "Fast-SCNN: Fast Semantic Segmentation Network".
