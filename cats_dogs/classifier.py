#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Union

import cv2
import numpy as np

from .dnn_api import ModelWrapper
from .utils import is_image


class CatDogClassifier:
    """
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
    """
    CLASS_NAMES = ['cat', 'dog']
    UNKNOWN_CLASS_NAME = 'unknown_class'
    UNSUPPORTED_FILE_MESSAGE = 'unsupported_file'

    def __init__(self,
                 model_path: str,
                 class_confidence_threshold: float):
        self._model = ModelWrapper.load(model_path)
        self._class_confidence_threshold = class_confidence_threshold

    def _get_class_name(self,
                        pred_probs: np.array) -> str:
        if len(pred_probs) != len(self.CLASS_NAMES):
            raise ValueError(f'Wrong model: number of classes is \
                             {len(pred_probs)} instead of \
                             {len(self.CLASS_NAMES)}.')
        most_probable = np.argmax(pred_probs)
        if pred_probs[most_probable] < self._class_confidence_threshold:
            return self.UNKNOWN_CLASS_NAME
        return self.CLASS_NAMES[most_probable]

    def predict_images(self,
                       images: List[np.array]) -> List[str]:
        """
        Predict cat/dog classes for list of OpenCV image objects.

        Parameters
        ----------
        images : list of OpenCV BGR image objects.

        Returns
        -------
        list[str] : "cat" or "dog" or "unknown_class" for each image.
        """
        pred_probs = self._model.predict(images)
        return list(map(self._get_class_name, pred_probs))

    def predict_file(self,
                     file: Union[str, bytes, bytearray]) -> str:
        """
        Predict cat/dog class for a single image file.

        Parameters
        ----------
        file : path to an image file, or bytes object.

        Returns
        -------
        str : "cat" or "dog" or "unknown_class" for an image,
        or "unsupported_file" if file is neither jpeg nor png.
        """
        if not is_image(file):
            return self.UNSUPPORTED_FILE_MESSAGE
        if isinstance(file, str):
            img = cv2.imread(file, cv2.IMREAD_COLOR)
        elif isinstance(file, (bytes, bytearray)):
            arr = np.frombuffer(file, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            raise TypeError('The argument must be str or bytes.')
        return self.predict_images([img])[0]

    def predict_dir(self,
                    dirpath: str) -> dict:
        """
        Predict cat/dog class for all image files in a given directory.

        Parameters
        ----------
        dirpath : path to directory with .jpg or .png files.

        Returns
        -------
        dict : {"<FILE NAME>": <IMAGE CLASS>},
        where <IMAGE CLASS> is "cat" or "dog"
        or "unknown_class" or "unsupported_file".
        """
        filenames = os.listdir(dirpath)
        filenames = filter(lambda f: os.path.isfile(os.path.join(dirpath, f)),
                           filenames)
        image_filenames = list(filter(lambda f: is_image(
            os.path.join(dirpath, f)),
                                      filenames))
        images = [cv2.imread(os.path.join(dirpath, f),
                             cv2.IMREAD_COLOR) \
                  for f in image_filenames]
        result = dict(zip(image_filenames, self.predict_images(images)))
        for filename in filenames:
            if not filename in image_filenames:
                result[filename] = self.UNSUPPORTED_FILE_MESSAGE
        return result
