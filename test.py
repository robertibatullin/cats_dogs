#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, os
from imutils import rotate
from src.classifier import CatDogClassifier
import argparse

parser = argparse.ArgumentParser(description='Testing cat\dog classifier.')
parser.add_argument('model_path', type=str, 
                    help='path to load DNN model')
parser.add_argument('image_path', type=str,  
                    help='''path to directory with "cats", "dogs" 
                    and optionally other subdirectories with images''')
parser.add_argument('--cct', type=float, default=0.9999,
                    help='''
                    Class confidence threshold. If all class probabilities 
                    are lower than this value, the object is classified 
                    as "unknown_class" (neither cat nor dog). By default 0.9999''')
parser.add_argument('--n_images', type=int, default=None,
                    help='''Max number of images to test in each subdirectory.
                    By default, all images are tested.''')
args = parser.parse_args()

test_dir = args.image_path
subdirs = [sbd for sbd in os.listdir(test_dir) \
           if os.path.isdir(os.path.join(test_dir, sbd))]

clf = CatDogClassifier(model_path = args.model_path,
              class_confidence_threshold = args.cct
              )

transform_funcs = {'ORIGINAL IMAGES': lambda x:x,
              'ROTATED BY 90 DEG': lambda x:rotate(x,90),
              'ROTATED BY 180 DEG': lambda x:rotate(x,180),
              'ROTATED BY 270 DEG': lambda x:rotate(x,270)}

for transform in transform_funcs:
    print(transform)
    transform_func = transform_funcs[transform]
    for subdir in subdirs:
        real_class = {'cats':'cat',
                      'dogs':'dog'}.get(subdir, 'unknown_class')
        accurate_predictions_count = 0
        filenames = os.listdir(os.path.join(test_dir,subdir))
        if args.n_images:
            filenames = filenames[:args.n_images]
        for filename in filenames:
            path = os.path.join(test_dir, subdir, filename)
            img = cv2.imread(path,
                             cv2.IMREAD_COLOR)
            img = transform_func(img)
            predicted_class = clf.predict_image(img)
            if real_class == predicted_class:
                accurate_predictions_count += 1
        accuracy_percent = int(100*accurate_predictions_count/len(filenames))
        print(real_class, ':',
              accurate_predictions_count, 'accurate predictions of',
              len(filenames),
              f'(accuracy = {accuracy_percent}%)')
                
                
    
    