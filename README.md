# CXV 
## Convolutional Xformers for Vision
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=convolutional-xformers-for-vision)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=convolutional-xformers-for-vision) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=convolutional-xformers-for-vision)


Vision transformers (ViTs) have found only limited practical use in processing images, in spite of their state-of-the-art accuracy on certain benchmarks. The reason for their limited use include their need for larger training datasets and more computational  resources compared to convolutional neural networks (CNNs), owing to the quadratic complexity of their self-attention mechanism. We propose a linear attention-convolution hybrid architecture -- Convolutional X-formers for Vision (CXV) -- to overcome these limitations. We replace the quadratic attention with linear attention mechanisms, such as Performer, Nyströmformer, and Linear Transformer, to reduce its GPU usage. Inductive prior for image data is provided by convolutional sub-layers, thereby eliminating the need for class token and positional embeddings used by the ViTs. CXV outperforms other architectures, token mixers (eg ConvMixer, FNet and MLP Mixer), transformer models (eg ViT, CCT, CvT and hybrid Xformers), and ResNets for image classification in scenarios with limited data and GPU resources.  

Models:
* CNV - Convolutional Nyströmformer for Vision
* CPV - Convolutional Performer for Vision
* CLTV - Convolutional Linear Transformer for Vision


Please cite the following papers if you are using the WaveMix model

```
@misc{https://doi.org/10.48550/arxiv.2201.10271,
  doi = {10.48550/ARXIV.2201.10271},
  
  url = {https://arxiv.org/abs/2201.10271},
  
  author = {Jeevan, Pranav and sethi, Amit},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.0; I.4.1; I.4.7; I.4.8; I.4.9; I.4.10; I.2.10; I.5.1; I.5.2; I.5.4},
  
  title = {Convolutional Xformers for Vision},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
