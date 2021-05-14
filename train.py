#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from src.dnn_api import ModelWrapper

parser = argparse.ArgumentParser(description='Train cat/doc classification model.')
parser.add_argument('model_path', type=str, 
                    help='path to save trained model')
parser.add_argument('train_dir', type=str,  
                    help='path to directory with train images')
parser.add_argument('val_dir', type=str,  
                    help='path to directory with validation images')
parser.add_argument('n_epochs', type=int,  
                    help='number of epochs for training')
parser.add_argument('--load', type=bool, default=True, 
                    help='load pre-trained model if True, initialize new if False. True by default')
args = parser.parse_args()

if args.load:
    model = ModelWrapper.load(args.model_path)
else:
    model = ModelWrapper.new()
model.prepare_train_val(args.train_dir, args.val_dir) 
model.train(n_epochs = args.n_epochs)
model.save(args.model_path)
