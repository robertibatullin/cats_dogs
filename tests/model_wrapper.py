#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import cv2

from cats_dogs.dnn_api import ModelWrapper


class TestModelWrapper(unittest.TestCase):

    def test_new(self):
        mw = ModelWrapper.new()
        self.assertIsInstance(mw, ModelWrapper)

    def test_load(self):
        mw = ModelWrapper.load('../model/model.hd5')
        self.assertIsInstance(mw, ModelWrapper)

    def test_predict(self):
        mw = ModelWrapper.load('../model/model.hd5')
        image = cv2.imread('../sample/cat.394.jpg')
        pred = mw.predict([image])
        self.assertEqual(pred.shape, (1, 2))
        self.assertGreaterEqual(pred[0][0], 0)
        self.assertLessEqual(pred[0][0], 1)


if __name__ == '__main__':
    unittest.main()
