# DASRGAN
Official PyTorch implementation of the paper Target-oriented Domain Adaptation for Infrared Image Super-Resolution.




## Introduction

Recent efforts have explored leveraging visible light images to enrich texture details in infrared (IR) super-resolution. However, such a direct adaptation scheme often results in a double-edged sword, enhancing texture while introducing noise and blurring artifacts. To address these challenges, we introduce Target-oriented Domain Adaptation SRGAN (DASRGAN), a novel framework designed to robustly adapt IR super-resolution models. DASRGAN is anchored on two bilateral collaborated components: 1) Texture-Oriented Adaptation (TOA) to refine texture details, and 2) Noise-Oriented Adaptation (NOA) aimed at reducing noise transfer. Specifically, TOA employs a specialized discriminator, with a prior extraction branch,  and Sobel-guided adversarial loss to align texture distributions. NOA incorporates noise adversarial loss to diverge generative and Gaussian noise pattern distributions in adversarial training.
Our comprehensive experiments validate the framework's efficacy. Comparative evaluations against existing methods across multiple benchmark datasets and upsampling factors reveal that DASRGAN not only sets state-of-the-art performance standards but also holds promise for real-world applications. 

## Coming soon....
