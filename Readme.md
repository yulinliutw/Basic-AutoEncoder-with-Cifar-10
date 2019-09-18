# Basic Autoencoder with CIFAR-10
---

This model can work on the Cifar-10, the model take the colour image as input, them its output try to reconstruct the image.
*Why the model do this work, you can google the Autoencoder, it may help you more understand this theory.*

It is authored by **YU LIN LIU**.

### Table of Contents
- <a href='#model-architecture'>Model Architecture</a>
- <a href='#installation'>Installation</a>
- <a href='#dataset'>Dataset</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#performance'>Performance</a>

## Model Architecture
---
The model content the encoder and decoder part, each module use the Conv layer, besides, the decoder use the ConvTranspose2d() layer to upsample the feature map.
In the model, by using the *LeakyReLU()* activation function, the model can more easy to train, it really prevent the *gradient vanishing problem*, so most activation function I use it, and only final activation function use the Relu() for trying to make input image back.
**For more detail of the model, you can check the autoencoder.py**
## Installation
---
- Install the [PyTorch](http://pytorch.org/) by selecting your environment on the website.
- Clone this repository.

## Dataset
---
#### CIFAR-10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
Current time, the pytorch library can directly provide this dataset, in my setting, I write the *load_data.py* it can load the data for training and evaluation, just check the *load_data.py*, *train.py* and *eval.py*, it can help you to know how to use it.  
**For the detail of this dataset in pytorch, you can check this** [link](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar).

## Training
---
- Open the train.py and check the argparse setting to understand the training parameters.
- Using the argparse for training parameter setting.
	* Note: we provide the pretrain weight in ./better_weight, you can load it by setting the *load_weight_dir* parameter.
- Start the training.
```Shell
python train.py
```	

## Evaluation
---
This action will show some result from model and the performane(average L1 loss per pixel) of the model. for the evaluation, I cut the 4000 data for validation, another testing data is for final evaluation.

- Open the eval.py and check the argparse setting to understand the evaluation parameters.
- Using the argparse for evaluation parameter setting.
- Start the evaluation.
```Shell
python eval.py
```	

## Performance
---
- I train this model about 600 epoch.
- Current performance(average L1 loss per pixel): 0.0153
- Some visualization result:

<p align="center">
<img src="https://github.com/yulinliutw/Basic-AutoEncoder-with-Cifar-10/blob/master/doc/result.png" alt=" " ></p>

