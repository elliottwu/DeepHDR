# [2017-07] Modifications for HDR: Shangzhe Wu
#   + License: MIT

import argparse
import os
import tensorflow as tf

from model import model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--c_dim', type=int, default=3)
parser.add_argument('--num_shots', type=int, default=3)
parser.add_argument('--dataset', type=str, default='dataset/tf_record/train.tfrecords')
parser.add_argument('--test_h', type=int, default=800)
parser.add_argument('--test_w', type=int, default=1200)
parser.add_argument('--save_freq', type=int, default=0)

args = parser.parse_args()

assert(os.path.exists(args.checkpoint_dir))

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model = model(sess, args, train=False)
    model.test(args)
