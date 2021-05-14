#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras import Input,Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2

IMAGE_SIZE = 224  

class ModelWrapper():
    '''
    Simplified wrapper for DNN classifiers. This implementation
    contains Tensorflow/Keras ResNet50 model. Others can be built around
    Torch etc. models. For external compatibility, they must contain the
    same methods.
    '''
    def __init__(self, model):
        self.model = model
        
    @classmethod
    def load(cls, path):
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
                     input_tensor = Input(shape = (IMAGE_SIZE, 
                                                   IMAGE_SIZE, 3)))
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
    
    def __preprocess_dataset(self, root):
        '''
        Helper function for creating Keras data generator from image folder.
    
        Parameters
        ----------
        root : path to directory containing subdirectories with image files. 
                Required structure:
                    root --- class1 --- image1
                       |          |---- image2
                       |
                       |---- class2 --- image3
    
        Returns
        -------
        Keras DataGenerator    
        '''
        datagen = image.ImageDataGenerator(
            preprocessing_function = preprocess_input)
        return datagen.flow_from_directory(
                root,
                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                batch_size = 32)

    def prepare_train_val(self, train_dir, val_dir):
        '''
        Creating train and val data generators from image folders

        Parameters
        ----------
        train_dir : path to directory containing subdirectories 
                    with image files for training.
        val_dir   : path to directory containing subdirectories 
                    with image files for validation.
        Required structure for both directories:
                    root --- class1 --- image1
                       |          |---- image2
                       |
                       |---- class2 --- image3
            
        Returns
        -------
        None.

        '''
        self.train_set = self.__preprocess_dataset(train_dir)
        self.val_set = self.__preprocess_dataset(val_dir)
    
    def train(self, n_epochs):
        '''
        Training model.

        Parameters
        ----------
        n_epochs : number of epochs.

        Raises
        ------
        AttributeError in case when train set and validation set are not assigned.

        Returns
        -------
        None.

        '''
        try:
            self.train_set, self.val_set
        except AttributeError:
            raise AttributeError('Train set and validation set not assigned. Call .prepare_train_val() method first.')
        self.model.compile(loss = "categorical_crossentropy", 
              optimizer = "adam",
              metrics = ["accuracy"])
        self.model.fit(
            self.train_set,
            steps_per_epoch = 200,
            epochs = n_epochs,
            validation_data = self.val_set)
    
    def save(self,path):
        '''
        Saving model.

        Parameters
        ----------
        path : path to save.

        Returns
        -------
        None.

        '''
        self.model.save(path)
        
    def predict(self, img):
        '''
        Predicting class probabilities for an image.

        Parameters
        ----------
        img : OpenCV image object.

        Returns
        -------
        pred_probs : Numpy array of class probabilities.

        '''
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        pred_probs = self.model.predict(np.expand_dims(img, axis=0))[0]
        return pred_probs
