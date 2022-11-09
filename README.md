# AIdea-Defect-Classifications-of-AOI
## Introduction
This project aims to implement a multi-input convolutional neural network referred to in a paper [[1]](https://www.graphyonline.com/archives/IJCSE/2018/IJCSE-137/) for Automated Optical Inspection (AOI) defect classificatons. However, in the project, pretrained Resnet50 models are used to replace CNNs designed in the original paper, and the output layers of the two Resnet50 networks are concatenated followed by the last linear layer. The accuracy of the model reaches 99.21% on the testing data provided by AIdea [[2]](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4), and the score is ranked 55th out of 432 groups on 8th of November in 2022. The total number of groups is 910, which means there are 478 groups fail to upload their solutions.
## Data description
### training data: 
2528 PNG images
### testing data: 
10142 PNG images
### Labels: 
0 : normal
1 : void
2 : horizontal defect
3 : vertical defect
4 : edge defect
5 : particle
