#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras import Input,Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2

class ModelWrapper():
    def __init__(self, model):
        self.model = model
        
    @classmethod
    def load(cls, path):
        model = load_model(path)
        return cls(model)
    
    @classmethod
    def new(cls):
        # Loading ResNet50 pre-trained on ImageNet 
        pre_trained = ResNet50(weights="imagenet", 
                     include_top=False, 
                     input_tensor=Input(shape=(224, 224, 3)))
        for layer in pre_trained.layers:
            layer.trainable = False
        # Adding trainable Dense layer
        top_layer = pre_trained.output
        top_layer = Flatten(name="flatten")(top_layer)
        top_layer = Dense(256, activation="relu")(top_layer)
        top_layer = Dense(2, activation="softmax")(top_layer)
        model = Model(inputs=pre_trained.input, 
                      outputs=top_layer)
        return cls(model)
    
    def train(self, 
              train_dir, 
              valid_dir,
              n_epochs):
        # Preparing datasets
        train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
        valid_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=32)
        valid_generator = valid_datagen.flow_from_directory(
                valid_dir,
                target_size=(224, 224),
                batch_size=32)
        #compiling and training
        self.model.compile(loss="categorical_crossentropy", 
              optimizer='adam',
              metrics=["accuracy"])
        self.model.fit(
            train_generator,
            steps_per_epoch=200,
            epochs=n_epochs,
            validation_data=valid_generator)
    
    def save(self,path):
        self.model.save(path)
        
    def predict(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))
        pred_probs = self.model.predict(np.expand_dims(image, axis=0))[0]
        return pred_probs

