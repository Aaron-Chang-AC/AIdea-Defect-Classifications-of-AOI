# AIdea-Defect-Classifications-of-AOI
## Introduction
This project aims to implement a multi-input convolutional neural network referred to in a paper [[1]](https://www.graphyonline.com/archives/IJCSE/2018/IJCSE-137/) for Automated Optical Inspection (AOI) defect classificatons. However, in the project, pretrained Resnet50 models are used to replace CNNs designed in the original paper (CNN2), and the output layers of the two Resnet50 [[2]] (https://arxiv.org/abs/1512.03385) networks are concatenated followed by the last linear layer. The accuracy of the model reaches 99.21% on the testing data provided by AIdea [[3]](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4), and the score was ranked 55th out of 432 groups on 8th of November in 2022. The total number of groups was 910, which meant there were 478 groups failed to upload their solutions.  
## Specifications
Platform: Google Colaboratory  
Language: Python 3.7.13  
Core Imported Modules: Pytorch, Torchvision, Transformers, Tqdm, etc.  
GPU: Tesla P100-PCIE-16GB  
## Data description
The data includes images of PNG format and labels ranging from 0 to 5.  
### training data: 
2528 PNG images
### testing data: 
10142 PNG images
### Labels: 
0 : normal,
1 : void,
2 : horizontal defect,
3 : vertical defect,
4 : edge defect,
5 : particle.
## Model
### Structure
The figure below illustrates how the model classifies each input image into 6 categories. First of all, an input image is duplicated (and rotated properly with a predefined probability), and one of the duplicated images is further sharpened as the second input tensor. Next, there are two Resnet50 models responsible for extracting features from the two images. After this, the tensors are concatenated before a linear layer and a softmax layer. Finally, the output tensor with the shape (batch_size, 6) is produced. For instance, when there is only one image, the output tensor has the shape (1, 6).
![Model](/display_images/model.png)
## Results
The settings of the hyperparameters are provided in the implementation code, and the scores are shown in the table below. For a comparison, the best result acquired by a VGG16 fine-tuned model is from another participant [[4]] (https://github.com/hcygeorge/aoi_defect_detection). The results implies that deep residual networks may perform better than VGG16 in terms of accuracy and number of trainable parameters (23M in Resnet50 and 138M in VGG16) on this task.
|Model |Accuracy|
|-----|--------|
|Resnet50 (one-input)     |99.01% |
|Resnet50 (multi-input)   |99.21% |
|VGG16 (one-input)  |99.0% |


