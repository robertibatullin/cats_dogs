#!/usr/bin/env python
# coding: utf-8

import argparse

from cats_dogs.classifier import CatDogClassifier
from cats_dogs.utils import prettyprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifying images in a given directory as cat or dog.')
    parser.add_argument('model_path', type=str,
                        help='path to load DNN model')
    parser.add_argument('image_path', type=str,
                        help='path to directory with images')
    parser.add_argument('-t', '--threshold', type=float, default=0.9999,
                        help='''
                        Class confidence threshold. If all class probabilities 
                        are lower than this value, the object is classified 
                        as "unknown_class" (neither cat nor dog). By default 0.9999''')
    args = parser.parse_args()

    clf = CatDogClassifier(model_path=args.model_path,
                           class_confidence_threshold=args.threshold
                           )
    prediction = clf.predict_dir(args.image_path)
    prettyprint(prediction)
