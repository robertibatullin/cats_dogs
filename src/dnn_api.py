#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
from tensorflow.keras import Input,Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class ModelWrapper():

    '''
    Simplified wrapper for DNN classifiers. This implementation
    contains Tensorflow/Keras ResNet50 model. Other ModelWrappers can be 
    built around Torch etc. models of any architecture. For external 
    compatibility, they must contain the same methods.
    '''

    IMAGE_SIZE = (224,224)  

    def __init__(self, model):
        self._model = model
        
    @classmethod
    def load(cls, path: str):
        '''
        Loading a trained model.
        Parameters
        ----------
        path : Path to model file or folder. 

        Returns
        -------
        ModelWrapper object.
        
        Example
        -------
        mw = ModelWrapper.load("model.hd5")

        '''
        model = load_model(path)
        return cls(model)
    
    @classmethod
    def new(cls):
        '''
        Creating a new ModelWrapper object.

        Returns
        -------
        ModelWrapper object.
        
        Example
        -------
        mw = ModelWrapper.new()
        '''
        # Loading ResNet50 model pre-trained on ImageNet dataset
        pre_trained = ResNet50(weights = "imagenet", 
                     include_top = False, 
                     input_tensor = Input(shape = (*cls.IMAGE_SIZE,3)))
        for layer in pre_trained.layers:
            layer.trainable = False
        # Adding trainable Dense layer
        top_layer = pre_trained.output
        top_layer = Flatten(name = "flatten")(top_layer)
        top_layer = Dense(256, activation = "relu")(top_layer)
        top_layer = Dense(2, activation = "softmax")(top_layer)
        model = Model(inputs = pre_trained.input, 
                      outputs = top_layer)
        return cls(model)
    
    def _preprocess_dataset(self, root: str):
        '''
        Helper function for creating Keras data generator from image folder.
    
        Parameters
        ----------
        root : path to directory containing subdirectories with image files. 
                Required structure:
                    root
                       |---class1
                       |      |---image1
                       |      |---image2
                       |---class2
                       |      |---image3
                       |      |---image4
    
        Returns
        -------
        Keras DataGenerator    
        '''
        datagen = image.ImageDataGenerator(
            preprocessing_function = preprocess_input)
        return datagen.flow_from_directory(
                root,
                target_size = self.IMAGE_SIZE,
                batch_size = 32)
    
    def train(self, train_dir: str, 
              val_dir: str, 
              n_epochs: int):
        '''
        Training model.

        Parameters
        ----------
        train_dir : path to directory containing subdirectories 
                    with image files for training.
        val_dir   : path to directory containing subdirectories 
                    with image files for validation.
        Required structure for both directories:
                    root
                       |---class1
                       |      |---image1
                       |      |---image2
                       |---class2
                       |      |---image3
                       |      |---image4
        n_epochs : number of epochs.

        Returns
        -------
        None.

        '''
        train_set = self._preprocess_dataset(train_dir)
        val_set = self._preprocess_dataset(val_dir)
        self._model.compile(loss = "categorical_crossentropy", 
              optimizer = "adam",
              metrics = ["accuracy"])
        self._model.fit(
            train_set,
            epochs = n_epochs,
            validation_data = val_set)
    
    def save(self, path:str):
        '''
        Saving model.

        Parameters
        ----------
        path : path to save.

        Returns
        -------
        None.

        '''
        self._model.save(path)
        
    def predict(self, imgs):
        '''
        Predicting class probabilities for several images.

        Parameters
        ----------
        imgs : list or other iterable of OpenCV BGR image objects.

        Returns
        -------
        pred_probs : Numpy array of shape (n_images, n_classes) of class probabilities.

        '''
        rgb_resized = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                              self.IMAGE_SIZE) for img in imgs]
        arr = np.array(rgb_resized)
        pred_probs = self._model.predict(arr)
        return pred_probs
