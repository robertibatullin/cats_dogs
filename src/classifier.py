#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, magic, cv2
import numpy as np
from .dnn_api import ModelWrapper

class CatDogClassifier():
    '''
    Object for classifying images as "cat" or "dog".       

    Initialization
    --------------
    clf = CatDogClassifier(model_path, class_confidence_threshold)

    Parameters
    ----------
    model_path : path to pre-trained DNN image classifier.
    class_confidence_threshold : If all class probabilities 
                    are lower than this value, the image is classified 
                    as "unknown_class" (neither cat nor dog).    

    '''
    def __init__(self,
                 model_path,
                 class_confidence_threshold):
        self.model = ModelWrapper.load(model_path)
        self.class_confidence_threshold = class_confidence_threshold
        
    def predict_image(self, img):
        '''
        Predict cat/dog class for an OpenCV image object.
        

        Parameters
        ----------
        img :  OpenCV BGR image object. 

        Returns
        -------
        str : "cat" or "dog" or "unknown_class".
        '''
        pred_probs = self.model.predict(img)
        if all([p<self.class_confidence_threshold for p in pred_probs]):
            return 'unknown_class'
        elif pred_probs[0] > pred_probs[1]:
            return 'cat'
        else:
            return 'dog'
    
    def predict_file(self,file):
        '''
        Predict cat/dog class for a single image file.

        Parameters
        ----------
        filepath : path to .jpg or .png file, or bytes object.

        Returns
        -------
        str : "cat" or "dog" or "unknown_class" 
              or "unsupported_file" if file mimetype 
              is neither jpeg nor png.
        '''
        if isinstance(file, str): # input is file path
            file_mimetype = magic.Magic(mime=True).from_file(file)
            if not file_mimetype in ('image/jpeg','image/png'):
                return 'unsupported_file'
            img = cv2.imread(file, cv2.IMREAD_COLOR)
        else: # input is bytes object
            file_mimetype = magic.Magic(mime=True).from_buffer(file)
            if not file_mimetype in ('image/jpeg','image/png'):
                return 'unsupported_file'
            arr = np.frombuffer(file, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.predict_image(img)
                
    def predict_dir(self,dirpath):
        '''
        Predict cat/dog class for all image file in a given directory.

        Parameters
        ----------
        dirpath : path to directory with .jpg or .png files.

        Returns
        -------
        dict : {"<FILE NAME>": <IMAGE CLASS>},
                where <IMAGE CLASS> is "cat" or "dog" 
                or "unknown_class" or "unsupported_file" 
                if file mimetype is neither jpeg nor png.       
        '''
        filenames = os.listdir(dirpath)
        return {filename: self.predict_file(os.path.join(dirpath,filename)) \
                for filename in filenames \
                if os.path.isfile(os.path.join(dirpath,filename))}
