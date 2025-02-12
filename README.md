# [DASRGAN](https://arxiv.org/abs/2311.08816)
Official PyTorch implementation of the paper Texture and Noise Dual Adaptation for Infrared Image Super-Resolution.


## Introduction

Recent efforts have explored leveraging visible light images to enrich texture details in infrared (IR) super-resolution. However, this direct adaptation approach often becomes a double-edged sword, as it improves texture at the cost of introducing noise and blurring artifacts. To address these challenges, we propose the Target-oriented Domain Adaptation SRGAN (DASRGAN), an innovative framework specifically engineered for robust IR super-resolution model adaptation. DASRGAN operates on the synergy of two key components: 1) Texture-Oriented Adaptation (TOA) to refine texture details meticulously, and 2) Noise-Oriented Adaptation (NOA), dedicated to minimizing noise transfer. Specifically, TOA uniquely integrates a specialized discriminator, incorporating a prior extraction branch, and employs a Sobel-guided adversarial loss to align texture distributions effectively. Concurrently, NOA utilizes a noise adversarial loss to distinctly separate the generative and Gaussian noise pattern distributions during adversarial training. Our extensive experiments confirm DASRGAN's superiority. Comparative analyses against leading methods across multiple benchmarks and upsampling factors reveal that DASRGAN sets new state-of-the-art performance standards.


## Requirements
> - Python 3.8, PyTorch >= 1.11
> - BasicSR 1.4.2
> - Platforms: Ubuntu 18.04, cuda-11

***You can also refer to this [Log](results/0131_DASRGAN_M3FD_x2_GitHub/test_0131_DASRGAN_M3FD_x2_GitHub_20240131_185144.log) for more info.***


## Installation
>  Clone the repo
```
git clone https://github.com/yongsongH/DASRGAN.git
```
> Install dependent packages
```
cd DASRGAN
```
```
pip install -r requirements.txt
```
> Install BasicSR
```
python setup.py develop
```
***You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation***

## Dataset prepare

Please check this [page](https://doi.org/10.6084/m9.figshare.28388330.v1).

## Model

Will be released after authorization.
