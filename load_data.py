##################################################
# Caution: returned LDRs are sorted by L-M-H, RGB
##################################################

import tensorflow as tf

GAMMA = 2.2 # LDR&HDR domain transform parameter
def LDR2HDR(img, expo): # input/output -1~1
    return (((img+1)/2.)**GAMMA / expo) *2.-1

def LDR2HDR_batch(imgs, expos): # input/output -1~1
    return tf.concat([LDR2HDR(tf.slice(imgs, [0,0,0], [-1, -1, 3]),expos[0]),
                      LDR2HDR(tf.slice(imgs, [0,0,3], [-1, -1, 3]),expos[1]),
                      LDR2HDR(tf.slice(imgs, [0,0,6], [-1, -1, 3]),expos[2])], 2)

def HDR2LDR(imgs, expo): # input/output -1~1
    return (tf.clip_by_value(((imgs+1)/2.*expo),0,1)**(1/GAMMA)) *2.-1

def transform_LDR(image, im_size=(256,256)):
    out = tf.cast(image, tf.float32)
    out = tf.image.resize_images(out, im_size)
    return out/127.5 - 1.

def transform_HDR(image, im_size=(256,256)):
    #out = tf.cast(image, tf.float32)
    out = tf.image.resize_images(image, im_size)
    return out*2. - 1.

def load_data(filename_queue, config):
    
    load_h, load_w, c = config.load_size, config.load_size, config.c_dim
    fine_h, fine_w = config.fine_size, config.fine_size
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'in_LDRs': tf.FixedLenFeature([], tf.string),
            'ref_LDRs': tf.FixedLenFeature([], tf.string),
            'in_exps': tf.FixedLenFeature([], tf.string),
            'ref_exps': tf.FixedLenFeature([], tf.string),
            'ref_HDR': tf.FixedLenFeature([], tf.string),
        })
    
    in_LDRs = tf.decode_raw(img_features['in_LDRs'], tf.uint8)
    ref_LDRs = tf.decode_raw(img_features['ref_LDRs'], tf.uint8)
    in_exps = tf.decode_raw(img_features['in_exps'], tf.float32)
    ref_exps = tf.decode_raw(img_features['ref_exps'], tf.float32)
    ref_HDR = tf.decode_raw(img_features['ref_HDR'], tf.float32)
    
    in_LDRs = tf.reshape(in_LDRs, [load_h, load_w, c*config.num_shots])
    ref_LDRs = tf.reshape(ref_LDRs, [load_h, load_w, c*config.num_shots])
    in_exps = tf.reshape(in_exps, [config.num_shots])
    ref_exps = tf.reshape(ref_exps, [config.num_shots])
    ref_HDR = tf.reshape(ref_HDR, [load_h, load_w, c])
    
    ######### distortions #########
    distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
    
    # flip horizontally
    in_LDRs = tf.cond(tf.less(distortions[0],0.5), lambda: tf.image.flip_left_right(in_LDRs), lambda: in_LDRs)
    ref_LDRs = tf.cond(tf.less(distortions[0],0.5), lambda: tf.image.flip_left_right(ref_LDRs), lambda: ref_LDRs)
    ref_HDR = tf.cond(tf.less(distortions[0],0.5), lambda: tf.image.flip_left_right(ref_HDR), lambda: ref_HDR)
    
    # rotate
    k = tf.cast(distortions[1]*4+0.5, tf.int32)
    in_LDRs = tf.image.rot90(in_LDRs, k)
    ref_LDRs = tf.image.rot90(ref_LDRs, k)
    ref_HDR = tf.image.rot90(ref_HDR, k)
    ######### distortions #########

    in_LDRs = transform_LDR(in_LDRs, [fine_h, fine_w])
    ref_LDRs = transform_LDR(ref_LDRs, [fine_h, fine_w])
    ref_HDR = transform_HDR(ref_HDR, [fine_h, fine_w])
    in_exps = 2.**in_exps
    ref_exps = 2.**ref_exps
    in_HDRs = LDR2HDR_batch(in_LDRs, in_exps)
    
    in_LDRs_batch, in_HDRs_batch, ref_LDRs_batch, ref_HDR_batch, in_exps_batch, ref_exps_batch = tf.train.shuffle_batch(
        [in_LDRs, in_HDRs, ref_LDRs, ref_HDR, in_exps, ref_exps],
        batch_size=config.batch_size,
        num_threads=2,
        capacity=256,
        min_after_dequeue=64)
    
    return in_LDRs_batch, in_HDRs_batch, ref_LDRs_batch, ref_HDR_batch, in_exps_batch, ref_exps_batch