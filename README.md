# Task---A-generalized-ML-framework

Author: Deniz Duğru

Date: 17.05.2022

Requirements: Python 3.8, Tensorflow 2.10, torch 1.11.0

  
  In this project, main approach was to create a ML framework that is compatible with different models and different frameworks (such as Pytorch, Keras). Models that I worked on and tested in this code are Pretrained melonomia model: https://www.kaggle.com/datasets/tirth27/melanoma-classification-kerasonnx-model-weight this was a Keras model and the other one is Alexnet: https://pytorch.org/hub/pytorch_vision_alexnet/. 
  
  
  I aimed to create a more generalized framework for both of them, however framework is compatible with only two of them. Structure is focused on logical operations in order to functionate seperately between the models. Class "GenPredicter" is taking many parameters, but first of all the user should create the model object and pass it to class with either image path or folder path. These two variables are must in order to use other functions. In order to preprocess image/s path should be passed, there are two different modes for preprocesing, user can preprocess only one image or can preprocess a whole folder and can resize, normalize data in his/her favor. In codes details are explained better. After the preprocess, we can predict with the preprocessed data and calculate loss and accuracy the control data must be passed into the function. There are more to improve and fine tune whole structure, we can make a new control system to define normalization type such as "mean" or "/255". A visualizaton function can be added in order to monitor accuracy and loss. For the second model which is pytorch there is no accuracy calculation, it can be added. Last but not least, framework can be improved to access and use other frameworks.

Here are some outputs of the predict() from models:

This output is with Keras model
![image](https://user-images.githubusercontent.com/63200146/168906716-59b997ec-bed8-4986-8cba-dd4d3ea63f8d.png)

This output is with Pytorch model
![image](https://user-images.githubusercontent.com/63200146/168907242-1f8068b3-5d16-41fb-9f95-ed3a743edf50.png)
