from sklearn import preprocessing
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision import models
import torch
from torchvision import transforms
import torch.nn.functional as nnf
import os
import torchvision
import torch.nn as nn
from keras.preprocessing import image
import keras
import torchmetrics

class GenPredicter:

    def __init__(self,model,img_path = None,folder_path=None,resizeInput=224,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] ): #initialized variables some of them are default
        self.model = model
        self.img_path = img_path
        self.resizeInput = resizeInput
        self.mean = mean
        self.std = std
        self.folder_path = folder_path
        #mean normalize ve 255 normalize parametre oluştur


    
    def preprocessImg(self):
    
        if "keras" in str(type(self.model)): #basic control if model is keras
            if self.img_path != None:#checking if passed image path
                image_k = tf.keras.preprocessing.image.load_img(self.img_path) #preprocessing single image
                image_k = image_k.resize((self.resizeInput, self.resizeInput)) #resizing based on input
                input_arr = tf.keras.preprocessing.image.img_to_array(image_k) #to array
                input_arr = np.array([input_arr]) 
                self.finalImg = input_arr
                
            elif self.folder_path != None: #checking if passed folder path
                images = []
                for img in os.listdir(self.folder_path):
                    img = tf.keras.preprocessing.image.load_img(self.folder_path + "/" + img) #preprocessing dataset
                    img = img.resize((self.resizeInput, self.resizeInput))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    images.append(img)
                images = np.vstack(images)
                self.finalImg = images
                
        
        elif "torch" in str(type(self.model)): #basic control if model is pytorch
            transform = transforms.Compose([
            transforms.Resize(size=self.resizeInput), #transformer for pytorch, all process can be done also via initialized variables
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( 
            self.mean,
            self.std 
            )])
            
            if self.img_path!=None: #for single image
                img = Image.open(self.img_path) #loading img with Pillow
                img_t = transform(img) #calling transformer
                batch_t = torch.unsqueeze(img_t,0) #most of the pytorch models accept batch as input
                self.finalImg = batch_t
                
            elif self.folder_path != None: #for multiple images
                filenames = [name for name in os.listdir(self.folder_path)] #an implementation to batch a dataset in a folder // if every data is image!!!
                batch_size = len(filenames)
                batch_t = torch.zeros(batch_size, 3, self.resizeInput, self.resizeInput, dtype=torch.uint8)
                for i, filename in enumerate(filenames):
                    img = Image.open(self.folder_path+"/"+filename)
                    img = img.convert("RGB")                    
                    img_t = transform(img)
                    batch_t[i] = img_t #collecting images into a batch
                batch_t = batch_t.float()
                self.finalImg = batch_t

        
        else:
            print("Model is unknown. Try with a keras or pytorch model.") #simply done to control
        
        return self.finalImg

        

    def predict(self):
        
        if "keras" in str(type(self.model)): #basic control mechanism
            self.predicted = self.model.predict(self.preprocessImg()) #preprocessing is done automatically, no obligation to run function seperately
            print(self.predicted)
        elif "torch" in str(type(self.model)):# basic control mechanism
            self.model.eval()
            self.output = self.model(self.preprocessImg()) #preprocessing is done automatically, no obligation to run function seperately
            prob = nnf.softmax(self.output, dim=1)
            top_p, top_class = prob.topk(1, dim = 1) #getting the best probability and the class it belongs to
            print(top_p, " ", top_class)
        else:
            print("Model is unknown. Try with a keras or pytorch model.") #simply done to control


    def calculateLoss(self,target):
        
        #labeling target images into a list        
        
        if "keras" in str(type(self.model)):
            cce = tf.keras.losses.CategoricalCrossentropy() #Calculating loss with buildin function, control data will be passed in function
            loss = cce(target, self.predicted).numpy()
            print(loss)

        elif "torch" in str(type(self.model)):
            self.output = torch.randn(3, 5, requires_grad=True)  #Loss is calculated with nn module, control data will be passed in function
            target = torch.empty(3, dtype=torch.long).random_(5)

            cross_entropy_loss = nn.CrossEntropyLoss()
            output_l = cross_entropy_loss(self.output, target)
            output_l.backward()
            print('input: ', self.output)
            print('target: ', target)
            print('output: ', output_l)
        else:
            print("Model is unknown. Try with a keras or pytorch model.") 
    
    def calculateMetrics(self,y_true):
        if "keras" in str(type(self.model)): 
            output = keras.metrics.categorical_accuracy(y_true, self.predicted) #Accuracy is calculated with buildin function
            print(output)
        elif "torch" in str(type(self.model)):
            pass
            

    
##### DEMO ##### 

#            PRETRAINED KERAS MODEL ON MEDICAL DATA

model = load_model('Model2_EffB4_No_meta.hdf5')
myModel = GenPredicter(model, folder_path="train/", resizeInput=380)
myModel.predict()


#            PRETRAINED PYTORCH MODEL ALEXNET
alexnet = models.alexnet(pretrained=True)
myModel2= GenPredicter(alexnet, folder_path="imagenet-samples")
myModel2.predict()

