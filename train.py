# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# [2017-07] Modifications for HDR: Shangzhe Wu
#   + License: MIT

import os
import scipy.misc
import numpy as np
import tensorflow as tf

from model import model


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("load_size", 250, "The size of images to be loaded [250]")
flags.DEFINE_integer("fine_size", 256, "The fine size of images [256]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("c_dim", 3, "The channal size of the images [3]")
flags.DEFINE_integer("num_shots", 3, "The number of exposure shots [3]")
flags.DEFINE_string("dataset", "dataset/tf_records", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the logs [logs]")
flags.DEFINE_integer("save_freq", 0, "Save frequency [0]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.69
with tf.Session(config=config) as sess:
    model = model(sess, config=FLAGS, train=True)
    model.train(FLAGS)
