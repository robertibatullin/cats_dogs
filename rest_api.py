#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Run this flask app on localhost and post image as follows:

     curl -X POST --data-binary @"<IMAGE FILE PATH>" localhost:5000/catdog

The response will be "cat", "dog", "unknown_class" or "unsupported_file".
'''

from flask import Flask, request
from src.classifier import CatDogClassifier

app = Flask(__name__)
clf = CatDogClassifier('model/model.hd5', 0.9999)

@app.route('/catdog', methods=['POST'])
def predict_cat_dog() -> str:
    '''
    View function receiving POST request with image file.

    Returns
    -------
    str : json string '{"class": <IMAGE CLASS>}'

    '''
    buff = request.get_data()
    pred = clf.predict_file(buff)
    return '{"class":"'+pred+'"}'

if __name__ == '__main__':
    app.run()
    