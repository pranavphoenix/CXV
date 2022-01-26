# CXV 
## Convolutional Xformers for Vision
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-tiny-imagenet-1)](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1?p=convolutional-xformers-for-vision)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=convolutional-xformers-for-vision) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-xformers-for-vision/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=convolutional-xformers-for-vision)


Vision transformers (ViTs) have found only limited practical use in processing images, in spite of their state-of-the-art accuracy on certain benchmarks. The reason for their limited use include their need for larger training datasets and more computational  resources compared to convolutional neural networks (CNNs), owing to the quadratic complexity of their self-attention mechanism. We propose a linear attention-convolution hybrid architecture -- Convolutional X-formers for Vision (CXV) -- to overcome these limitations. We replace the quadratic attention with linear attention mechanisms, such as Performer, Nyströmformer, and Linear Transformer, to reduce its GPU usage. Inductive prior for image data is provided by convolutional sub-layers, thereby eliminating the need for class token and positional embeddings used by the ViTs. CXV outperforms other architectures, token mixers (eg ConvMixer, FNet and MLP Mixer), transformer models (eg ViT, CCT, CvT and hybrid Xformers), and ResNets for image classification in scenarios with limited data and GPU resources.  

Models:
* CNV - Convolutional Nyströmformer for Vision
* CPV - Convolutional Performer for Vision
* CLTV - Convolutional Linear Transformer for Vision

