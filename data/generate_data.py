#!/usr/bin/env python3
import os
import random
import cv2
import sys
import csv
import time
import argparse
import numpy as np
from glob import glob

fl_path = os.path.abspath('.')
print(fl_path)

xray_path0 = os.path.join(fl_path, "chest_xray", "train", "NORMAL")

xray_path1 = os.path.join(fl_path, "chest_xray", "train", "PNEUMONIA")

images0 = glob(os.path.join(xray_path0, "*.jpeg"))
images1 = glob(os.path.join(xray_path1, "*.jpeg"))

test_path0 = os.path.join(fl_path, "chest_xray", "train", "NORMAL")

test_path1 = os.path.join(fl_path, "chest_xray", "train", "PNEUMONIA")

images_test0 = glob(os.path.join(test_path0, "*.jpeg"))
images_test1 = glob(os.path.join(test_path1, "*.jpeg"))

WIDTH = 150
HEIGHT = 150

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(0,199):
    if i%2==0:
        img = cv2.imread(random.sample(images0,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(0)
    else:
        img = cv2.imread(random.sample(images1,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(1)

for i in range(1,5000):
    if i%2==0:
        img = cv2.imread(random.sample(images_test0,1)[0])

        x_test.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_test.append(0)
    else:
        img = cv2.imread(random.sample(images1,1)[0])

        x_test.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_test.append(1)

name_file = 'data_party' + str(0) + '.npz'
name_file = os.path.join(fl_path, name_file)
np.savez(name_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

x_train = []
y_train = []
for i in range(0,199):
    if i%3==0:
        img = cv2.imread(random.sample(images0,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(0)
    else:
        img = cv2.imread(random.sample(images1,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(1)

name_file = 'data_party' + str(1) + '.npz'
name_file = os.path.join(fl_path, name_file)
np.savez(name_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

x_train = []
y_train = []
for i in range(0,199):
    if i%2==0:
        img = cv2.imread(random.sample(images0,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(0)
    else:
        img = cv2.imread(random.sample(images1,1)[0])

        x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_train.append(1)

name_file = 'data_party' + str(2) + '.npz'
name_file = os.path.join(fl_path, name_file)
np.savez(name_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)




