##################################################
# Caution: returned LDRs are sorted by L-M-H, RGB
##################################################

import tensorflow as tf
import numpy as np
import os
import glob
import cv2
from skimage.measure import compare_ssim as ssim


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


data_dir = './train'
input_name = "input_*_aligned.tif"
ref_name = "ref_*_aligned.tif"
input_exp_name = "input_exp.txt"
ref_exp_name = "ref_exp.txt"
ref_HDR_path = "ref_hdr_aligned.hdr"

scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, scene_dir))]
scene_dirs = sorted(scene_dirs)
num_scenes = len(scene_dirs)

out_dir = './tf_records/256_64_tfrecords'
patch_size = 256
patch_stride = 64
num_shots = 3 # number of exposure shots per scene
batch_size = 20

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

count = 0
cur_writing_path = os.path.join(out_dir, "train_{:d}_{:04d}.tfrecords".format(patch_stride, 0))
writer = tf.python_io.TFRecordWriter(cur_writing_path)

for i, scene_dir in enumerate(scene_dirs):
    if (i%10 == 0):
        print('%d/%d' %(i, num_scenes))
    
    # read images
    try:
        # assume the exposure increases with file name
        cur_dir = os.path.join(data_dir, scene_dir)
        in_LDR_paths = sorted(glob.glob(os.path.join(cur_dir, input_name)))
        tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
        h, w, c = tmp_img.shape

        in_LDRs = np.zeros((h, w, c*num_shots)) # stack 3 images along channel
        for j, in_LDR_path in enumerate(in_LDR_paths):
            in_LDRs[:,:,j*c:(j+1)*c] = cv2.imread(in_LDR_path).astype(np.uint8)
        in_LDRs = in_LDRs.astype(np.uint8)

        in_exps_path = os.path.join(cur_dir, input_exp_name)
        in_exps = np.array(open(in_exps_path).read().split('\n')[:num_shots]).astype(np.float32)

        ref_LDR_paths = sorted(glob.glob(os.path.join(cur_dir, ref_name)))
        ref_LDRs = np.zeros((h, w, c*num_shots)) # stack 3 images along channel
        for j, ref_LDR_path in enumerate(ref_LDR_paths):
            ref_LDRs[:,:,j*c:(j+1)*c] = cv2.imread(ref_LDR_path).astype(np.uint8)
        ref_LDRs = ref_LDRs.astype(np.uint8)
        
        ref_HDR = cv2.imread(os.path.join(cur_dir, ref_HDR_path),-1).astype(np.float32) # read raw values

        ref_exps_path = os.path.join(cur_dir, ref_exp_name)
        ref_exps = np.array(open(ref_exps_path).read().split('\n')[:num_shots]).astype(np.float32)
    
    except IOError as e:
        print('Could not read:', scene_dir)
        print('error: %s' %e)
        print('Skip it!\n')
        continue

    def write_example(h1, h2, w1, w2):
        global count
        global writer
        
        cur_batch_index = count // batch_size

        if count%batch_size == 0:
            writer.close()
            cur_writing_path = os.path.join(out_dir, "train_{:d}_{:04d}.tfrecords".format(patch_stride, cur_batch_index))
            writer = tf.python_io.TFRecordWriter(cur_writing_path)

        # reverting them from BGR to RGB
        in_LDRs_patch = in_LDRs[h1:h2, w1:w2, :]
        in_LDRs_patch_1 = in_LDRs_patch[:, :, 2::-1]
        in_LDRs_patch_2 = in_LDRs_patch[:, :, 5:2:-1]
        in_LDRs_patch_3 = in_LDRs_patch[:, :, 8:5:-1]
        in_LDRs_patch = np.concatenate([in_LDRs_patch_1, in_LDRs_patch_2, in_LDRs_patch_3], axis=2)

        ref_LDRs_patch = ref_LDRs[h1:h2, w1:w2, :]
        ref_LDRs_patch_1 = ref_LDRs_patch[:, :, 2::-1]
        ref_LDRs_patch_2 = ref_LDRs_patch[:, :, 5:2:-1]
        ref_LDRs_patch_3 = ref_LDRs_patch[:, :, 8:5:-1]
        ref_LDRs_patch = np.concatenate([ref_LDRs_patch_1, ref_LDRs_patch_2, ref_LDRs_patch_3], axis=2)

        ref_HDR_patch = ref_HDR[h1:h2, w1:w2, ::-1]

        count += 1

        # create example
        example = tf.train.Example(features=tf.train.Features(feature={
            'in_LDRs':bytes_feature(in_LDRs_patch.tostring()),
            'ref_LDRs': bytes_feature(ref_LDRs_patch.tostring()),
            'in_exps': bytes_feature(in_exps.tostring()),
            'ref_exps': bytes_feature(ref_exps.tostring()),
            'ref_HDR': bytes_feature(ref_HDR_patch.tostring()),
            }))
        writer.write(example.SerializeToString())

    # generate patches
    for h_ in range(0, h-patch_size+1, patch_stride):
        for w_ in range(0, w-patch_size+1, patch_stride):
            write_example(h_, h_+patch_size, w_, w_+patch_size)

    # deal with border patch
    if h%patch_size:
        for w_ in range(0, w-patch_size+1, patch_stride):
            write_example(h-patch_size, h, w_, w_+patch_size)

    if w%patch_size:
        for h_ in range(0, h-patch_size+1, patch_stride):
            write_example(h_, h_+patch_size, w-patch_size, w)

    if w%patch_size and h%patch_size :
        write_example(h-patch_size, h, w-patch_size, w)

writer.close()
print("Finished!\nTotal number of patches:", count)
