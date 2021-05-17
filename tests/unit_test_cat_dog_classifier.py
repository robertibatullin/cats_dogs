#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import cv2

from cats_dogs.classifier import CatDogClassifier 


class TestCatDogClassifier(unittest.TestCase):

    def test_predict_images(self):
        clf = CatDogClassifier('../model/model.hd5', 0.9999)
        images = [cv2.imread('../sample/cat.394.jpg'),
                  cv2.imread('../sample/dog.1402.jpg')]
        pred = clf.predict_images(images)
        self.assertEqual(len(pred),2)
        self.assertIsInstance(pred[0], str)
        self.assertIsInstance(pred[1], str)
        self.assertIn(pred[0], ['cat','dog','unknown_class'])
        self.assertIn(pred[1], ['cat','dog','unknown_class'])

    def test_predict_file_by_path(self):
        clf = CatDogClassifier('../model/model.hd5', 0.9999)
        file = '../sample/cat.394.jpg'
        pred = clf.predict_file(file)
        self.assertIsInstance(pred, str)
        self.assertIn(pred, ['cat','dog','unknown_class'])
        
    def test_predict_file_by_bytes(self):
        clf = CatDogClassifier('../model/model.hd5', 0.9999)
        file = open('../sample/cat.394.jpg','rb').read()
        pred = clf.predict_file(file)
        self.assertIsInstance(pred, str)
        self.assertIn(pred, ['cat','dog','unknown_class'])

    def test_predict_dir(self):
        clf = CatDogClassifier('../model/model.hd5', 0.9999)
        path = '../sample'
        pred = clf.predict_dir(path)
        self.assertIsInstance(pred, dict)
        first_value = list(pred.values())[0]
        self.assertIn(first_value, ['cat','dog','unknown_class'])

if __name__ == '__main__':
    unittest.main()
    
