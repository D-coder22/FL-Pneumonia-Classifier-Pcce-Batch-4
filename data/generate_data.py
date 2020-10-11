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

WIDTH = 224
HEIGHT = 224

x_train = []
y_train = []
x_test = []
y_test = []
# for i in range(0,199):
#     if i%2==0:
#         img = cv2.imread(random.sample(images0,1)[0])

#         x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
#         y_train.append(0)
#     else:
#         img = cv2.imread(random.sample(images1,1)[0])

#         x_train.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
#         y_train.append(1)

for i in range(1,5000):
    if i%2==0:
        img = cv2.imread(random.sample(images_test0,1)[0])

        x_test.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_test.append(0)
    else:
        img = cv2.imread(random.sample(images1,1)[0])

        x_test.append(cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        y_test.append(1)

# name_file = 'data_party' + str(0) + '.npz'
# name_file = os.path.join(fl_path, name_file)
# np.savez(name_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

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




# if fl_path not in sys.path:
#     sys.path.append(fl_path)

# from ibmfl.util.datasets import load_xray
# from classifiers.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
#     STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC


# def setup_parser():
#     """
#     Sets up the parser for Python script

#     :return: a command line parser
#     :rtype: argparse.ArgumentParser
#     """
#     p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
#     p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
#                    type=int, required=True)
#     p.add_argument("--dataset", "-d", choices=FL_DATASETS,
#                    help=DATASET_DESC, required=True)
#     p.add_argument("--data_path", "-p", help=PATH_DESC,
#                    default=os.path.join("examples", "data"))
#     p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
#                    nargs="+", type=int, required=True)
#     p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
#     p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
#     p.add_argument("--name", help=NAME_DESC)
#     return p


# def print_statistics(i, x_test_pi, x_train_pi, nb_labels, y_train_pi):
#     print('Party_', i)
#     print('nb_x_train: ', np.shape(x_train_pi),
#           'nb_x_test: ', np.shape(x_test_pi))
#     for l in range(nb_labels):
#         print('* Label ', l, ' samples: ', (y_train_pi == l).sum())



# def save_xray_party_data(nb_dp_per_party, should_stratify, party_folder):
#     """
#     Saves XRAY party data

#     :param nb_dp_per_party: the number of data points each party should have
#     :type nb_dp_per_party: `list[int]`
#     :param should_stratify: True if data should be assigned proportional to source class distributions
#     :type should_stratify: `bool`
#     :param party_folder: folder to save party data
#     :type party_folder: `str`
#     """
#     dataset_path = os.path.join("classifiers", "datasets")
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)
#     (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_path)
#     labels, train_counts = np.unique(y_train, return_counts=True)
#     te_labels, test_counts = np.unique(y_test, return_counts=True)
#     if np.all(np.isin(labels, te_labels)):
#         print("Warning: test set and train set contain different labels")

#     num_train = np.shape(y_train)[0]
#     num_test = np.shape(y_test)[0]
#     num_labels = np.shape(np.unique(y_test))[0]
#     nb_parties = len(nb_dp_per_party)

#     if should_stratify:
#         # Sample according to source label distribution
#         train_probs = {
#             label: train_counts[label] / float(num_train) for label in labels}
#         test_probs = {label: test_counts[label] /
#                       float(num_test) for label in te_labels}
#     else:
#         # Sample uniformly
#         train_probs = {label: 1.0 / len(labels) for label in labels}
#         test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

#     for idx, dp in enumerate(nb_dp_per_party):
#         train_p = np.array([train_probs[y_train[idx]]
#                             for idx in range(num_train)])
#         train_p /= np.sum(train_p)
#         train_indices = np.random.choice(num_train, dp, p=train_p)
#         test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
#         test_p /= np.sum(test_p)

#         # Split test evenly
#         test_indices = np.random.choice(
#             num_test, int(num_test / nb_parties), p=test_p)

#         x_train_pi = x_train[train_indices]
#         y_train_pi = y_train[train_indices]
#         x_test_pi = x_test[test_indices]
#         y_test_pi = y_test[test_indices]

#         # Now put it all in an npz
#         name_file = 'data_party' + str(idx) + '.npz'
#         name_file = os.path.join(party_folder, name_file)
#         np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
#                  x_test=x_test_pi, y_test=y_test_pi)

#         print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

#         print('Finished! :) Data saved in ', party_folder)



# if __name__ == '__main__':
#     # Parse command line options
#     parser = setup_parser()
#     args = parser.parse_args()

#     # Collect arguments
#     num_parties = args.num_parties
#     dataset = args.dataset
#     data_path = args.data_path
#     points_per_party = args.points_per_party
#     stratify = args.stratify
#     create_new = args.create_new
#     exp_name = args.name

#     # Check for errors
#     if len(points_per_party) == 1:
#         points_per_party = [points_per_party[0] for _ in range(num_parties)]
#     elif len(points_per_party) != num_parties:
#         parser.error(PER_PARTY_ERR)

#     # Create folder to save party data
#     folder = os.path.join("classifiers", "data")
#     strat = 'balanced' if stratify else 'random'

#     if create_new:
#         folder = os.path.join(folder, exp_name if exp_name else str(
#             int(time.time())) + '_' + strat)
#     else:
#         folder = os.path.join(folder, dataset, strat)

#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     else:
#         # clear folder of old data
#         for f_name in os.listdir(folder):
#             f_path = os.path.join(folder, f_name)
#             if os.path.isfile(f_path):
#                 os.unlink(f_path)

#     # Save new files
#     if args.dataset == 'xray':
#         save_xray_party_data(points_per_party, stratify, folder)
    
