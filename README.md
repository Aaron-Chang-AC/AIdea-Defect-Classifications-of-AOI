# AIdea-Defect-Classifications-of-AOI
## Introduction
This project aims to implement a multi-input convolutional neural network referred to in a paper [[1]](https://www.graphyonline.com/archives/IJCSE/2018/IJCSE-137/) for Automated Optical Inspection (AOI) defect classifications. However, in the project, pretrained models are used to replace CNNs designed in the original paper (CNN2), and the output layers of the two models (which can be Resnet50 [[2]](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) networks, Vision Transformers [[3]](https://arxiv.org/pdf/2010.11929.pdf), or ViTs with Masked Autoencoders [[4]](https://arxiv.org/pdf/2111.06377v2.pdf)) are concatenated followed by the last linear layer. The accuracy of the Resnet50 model reaches 99.21% on the testing data provided by AIdea [[5]](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4). In addition, a Vision Transformer (ViT) version is also implemented , and the accuracy score from ViT was ranked 6th (99.70%) which was the same as that of the 5th group. It is shown in the results that a ViTMAE multi-input model (99.31%) cannot outperform its ViT counterpart. However, the best accuracy score 99.75% can be obtained with majority voting which combines the performances of a ViT single-input model, the multi-input ViT model and the ViTMAE multi-input model. The score was ranked 5th (the same as that of the 4th group) out of 432 groups on 17th of November in 2022. The total number of groups was 922, which meant there were 490 groups failed to upload their solutions.  
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
The figure below illustrates how the model classifies each input image into 6 categories. First of all, an input image is duplicated (and rotated properly with a predefined probability), and one of the duplicated images is further sharpened to become the second input tensor. Next, there are two pretrained models (ViT in the example) responsible for extracting features from the two images. After this, the tensors are concatenated before a linear layer and a softmax layer. Finally, the output tensor with the shape (batch_size, 6) is produced. For instance, when there is only one image, the output tensor has the shape (1, 6).
![Model](/display_images/model.png)
## Results
The settings of the hyperparameters are provided in the implementation code, and the scores calculated by AIdea are shown in the table below. For a comparison, the best result acquired by a VGG16 fine-tuned model is from another participant [[6]](https://github.com/hcygeorge/aoi_defect_detection). The results imply that deep residual networks may perform better than VGG16 in terms of the accuracy and the number of trainable parameters (23M in Resnet50 and 138M in VGG16) on this task. Furthermore, ViT outperforms others with the highest accuracy score, which indicates that self-attention-based mechanisms may perform better than other state-of-the-art CNNs.
|Model |Accuracy Score|
|-----|--------|
|VGG16 (one-input)  |99.0% |
|Resnet50 (one-input)     |99.01% |
|Resnet50 (multi-input)   |99.21% |
|ViT-base (one-input)   |99.58% |
|ViT-base (multi-input)   |99.70% |
|ViTMAE-base (multi-input)   |99.31% |
|Ensemble (ViT-one-input, ViT-multi-input, ViTMAE-multi-input)   |99.75% |  

![rank](/display_images/rank.PNG)
## References
[[1] T. Shiina, Y. Iwahori, and B. Kijsirikul, “Defect classification of electronic circuit board using multi-input convolutional neural network,” Int. J. Comput. Softw. Eng., vol. 3, no. 2, pp. 2456–4451, 2018.](https://www.graphyonline.com/archives/IJCSE/2018/IJCSE-137/)  
[[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)  
[[3] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.](https://arxiv.org/abs/2010.11929)  
[[4] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16000–16009.](https://arxiv.org/abs/2111.06377)  
[[5] AIdea](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)  
[[6] A VGG16 fine-tuned model from another participant](https://github.com/hcygeorge/aoi_defect_detection)  
