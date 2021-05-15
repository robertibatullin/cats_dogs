#!/usr/bin/env python
# coding: utf-8

import argparse
from src.classifier import CatDogClassifier

def prettyprint(dct: dict):
    '''
    Utility for printing CatDogClassifier output

    Parameters
    ----------
    dct : dict of {"<FILE NAME>": <IMAGE CLASS>}

    Returns
    -------
    None.

    '''
    max_key_len = max([len(str(key)) for key in dct.keys()])
    for key in sorted(dct.keys()):
        print(key,' '*(max_key_len-len(str(key)))+':',dct[key])
           
parser = argparse.ArgumentParser(description='Classifying images in a given directory as cat or dog.')
parser.add_argument('model_path', type=str, 
                    help='path to load DNN model')
parser.add_argument('image_path', type=str,  
                    help='path to directory with images')
parser.add_argument('--cct', type=float, default=0.9999,
                    help='''
                    Class confidence threshold. If all class probabilities 
                    are lower than this value, the object is classified 
                    as "unknown_class" (neither cat nor dog). By default 0.9999''')
args = parser.parse_args()

clf = CatDogClassifier(model_path = args.model_path,
              class_confidence_threshold = args.cct
              )
prediction = clf.predict_dir(args.image_path)
prettyprint(prediction)
