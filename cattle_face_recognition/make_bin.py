#coding:utf-8
import random

import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
random_seed = 2048
random.seed(random_seed)
np.random.seed(random_seed)
def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename) as f:
        for line in f:
            pair = line.strip().split(' ')
            pairs.append(pair)
    random.shuffle(pairs)
    return np.array(pairs)


def get_paths(pairs):
    # print(pairs)
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    cnt = 1
    for pair in pairs:
        print(pair)
        path0 = pair[0]
        path1 = pair[1]

        if '1' == pair[2]:
            issame = True
        else:
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
        cnt += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package  images')
    # general
    parser.add_argument('--data-dir', default='/mnt/workspace/cow_face/code/last_arcface/datasets/pyramid/test_cow', help='')
    parser.add_argument('--image-size', type=str, default='112,112', help='')
    parser.add_argument('--output', default='/mnt/workspace/cow_face/code/last_arcface/datasets/pyramid/train_cow/test_cow.bin', help='path to save.')
    parser.add_argument('--txtfile', default='/mnt/workspace/cow_face/code/last_arcface/datasets/pyramid/pairs.txt', help='txtfile path.')
    args = parser.parse_args()
    image_size = [int(x) for x in args.image_size.split(',')]
    img_pairs = read_pairs(args.txtfile)
    img_paths, issame_list = get_paths(img_pairs)
    img_bins = []
    i = 0
    for path in img_paths:
        with open(path, 'rb') as fin:
            _bin = fin.read()
            img_bins.append(_bin)
            i += 1
    with open(args.output, 'wb') as f:
        pickle.dump((img_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
