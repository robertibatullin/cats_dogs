#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from src.utils import is_image 


class TestIsImage(unittest.TestCase):
    
    def test_is_image_by_path(self):
        image_file = 'sample/cat.394.jpg'
        not_image_file = 'sample/notimage.1.txt'
        self.assertEqual(is_image(image_file), True)
        self.assertEqual(is_image(not_image_file), False)
        
    def test_is_image_by_bytes(self):
        image_file = open('sample/cat.394.jpg','rb').read()
        not_image_file = open('sample/notimage.1.txt','rb').read()
        self.assertEqual(is_image(image_file), True)
        self.assertEqual(is_image(not_image_file), False)
        

if __name__ == '__main__':
    unittest.main()
    