# dataset_gradcam_plus_plus

Grad-CAM ++ Algorithms which gives a better  Visual Explainations for the decisions made by Deep Learning Model Using Tesorflow-2. However, the explanation is give at local level. Our library give a dataset level explanability for 1d signal. 
For bigger dataset (>5000) , we recommand to split the dataset into numerous batchs of 5000 samples.

# Installation

> pip install dataset_gradcam_plus_plus
>                 or
> cd ./dataset_gradcam_plus_plus  python setup.py install

# Local classification explaination

from dataset_gradcam_plus_plus import local_features_importances

local_features_importances(model,data,time,layer_name,label)
Arguments:
> - model : The model trained
> - data : The data sample
> - time : The time range. The user can create a linspace vector and use it as the time range.
> - layer_name : The last layer of the feature extraction part of the model. Usually, it is the last layer before the Flattening operation.
> - label : The label output list

![Alt text](https://github.com/marcusnk237/dataset_gradcam_plus_plus/blob/main/results/gradcam_plus_plus_1d.png)

# Dataset level feature relevance

from dataset_gradcam_plus_plus import dataset_features_importances

dataset_features_importances(model,datas,layer_name,feature_names,flag=True)

Arguments:
> - model : The model trained
> - datas : the dataset. Due of Hardware limitations, the max size of datas is 5000 samples.
> - layer_name : The last layer of the feature extraction part of the model. Usually, it is the last layer before the Flattening operation.
> - feature_names : The feature list
> - flag : True, return the global feature importance plot; False, return the feature importance frequency and the corresponding features

![Alt text](https://github.com/marcusnk237/dataset_gradcam_plus_plus/blob/main/results/dataset_level_feature_importance.jpg)

# Sources : 
The original Grad-CAM ++ publication :https://arxiv.org/pdf/1710.11063.pdf

# License
This project is Licensed under the MIT License.
