import tensorflow as tf
import scipy as sp
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import random

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers


def ASPP(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.') 
    #
    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2*rate for rate in atrous_rates]
    #
    #why do we need arg_scope of resnet_v2
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
            inputs_size = tf.shape(inputs)[1:3]
            conv_11 = layers_lib.conv2d(inputs, depth, [1, 1], stride = 1, scope = "conv_1x1")
            conv_33_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride = 1, rate = atrous_rates[0], scope = 'conv_3x3_1')
            conv_33_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride = 1, rate = atrous_rates[1], scope = 'conv_3x3_2')
            conv_33_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride = 1, rate = atrous_rates[2], scope = 'conv_3x3_3')
            #
            with tf.variable_scope("image_level_features"):
                image_level_features = tf.reduce_mean(inputs, [1,2], name = 'global_average_pooling', keepdims = True)
                image_level_features = layers_lib.conv2d(image_level_features, depth, [1,1], stride = 1, scope = 'conv_1x1')
                image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            #
            net = tf.concat([conv_11, conv_33_1, conv_33_2, conv_33_3, image_level_features], axis = 3, name = 'concat')
            net = layers_lib.conv2d(net, depth, [1, 1], stride = 1, scope = 'conv_1x1_concat')
            #
            return net

def DeepLabNet(input_batch, is_training, num_classes, output_stride = 16, batch_norm_decay = 0.9997, backbone = 'resnet_v2_101'):
    #Use channels_first to boost on GPU
    #Deeplab V3+ with resnet as backbone
    inputs_size = tf.shape(input_batch)[1:3]
    with tf.variable_scope('deeplab'):
        #ResNet as the encoder
        with tf.variable_scope('encoder'):
            if backbone == 'resnet_v2_50':
                base_model = resnet_v2.resnet_v2_50
            else:
                base_model = resnet_v2.resnet_v2_101
            #
            #Implement tensorflow resnetV2
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                logits, end_points = base_model(input_batch,
                                                num_classes=None,
                                                is_training=is_training,
                                                global_pool=False,
                                                output_stride=output_stride)
        #ASPP in the middle layers
        with tf.variable_scope('aspp'):
            net = end_points['deeplab/encoder/' + backbone + '/block4']
            encoder_output = ASPP(net, output_stride, batch_norm_decay, is_training)
        #
        #Decoder
        with tf.variable_scope('decoder'):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay = batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    with tf.variable_scope("low_level_features"):
                        low_level_features = end_points['deeplab/encoder/' + backbone + '/block1/unit_3/bottleneck_v2/conv1']
                        low_level_features = layers_lib.conv2d(low_level_features, 48, [1,1], stride = 1, scope = 'conv_1x1')
                        low_level_features_size = tf.shape(low_level_features)[1:3]
                    with tf.variable_scope("upsampling_logits"):
                        net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name = 'upsample_1')
                        net = tf.concat([net, low_level_features], axis = 3, name = 'concat')
                        net = layers_lib.conv2d(net, 256, [3,3], stride = 1, scope = 'conv_3x3_1')
                        net = layers_lib.conv2d(net, 256, [3,3], stride = 1, scope = 'conv_3x3_2')
                        net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'conv_1x1')
                        logits = tf.image.resize_bilinear(net, inputs_size, name = 'upsample_2')
    #
    return logits



def load_data():
image_path = r"Z:\Kaggle\Steel\severstal-steel-defect-detection\train_sample"
filelist = os.listdir(image_path)

label_excel = pd.read_csv(r"Z:\Kaggle\Steel\severstal-steel-defect-detection\train.csv")
label_excel = label_excel.fillna("")

label = dict()
feature = dict()
for file in filelist:
    print(file)
    im = sp.misc.imread(image_path + '\\' + file, flatten = True)
    width, height = im.shape
    feature[file] = im
    label[file] = np.zeros([width, height])
    for i in range(1, 5):
        positions = str(label_excel[label_excel.ImageId_ClassId == file + "_" + str(i)].EncodedPixels.iloc[0]).split(' ')
        if positions!= [""]:
            for j in range(0, len(positions), 2):
                for h in range(int(positions[j]) - 1, int(positions[j]) - 1 + int(positions[j+1])):
                    label[file][h%width][h//width] = i






def main():




    input_batch = tf.placeholder(tf.float32, shape = (None,None,None,1), name = 'input_batch')
    label_batch = tf.placeholder(tf.int32, shape = (None,None,None), name = 'label_batch')
    is_training = tf.placeholder(tf.bool, name = 'is_training')
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(7e-3, tf.cast(global_step, tf.int32), 30000, 1e-6, power=0.9)



    logits = DeepLabNet(input_batch, is_training, 5, backbone = 'resnet_v2_50')
    logits_by_num_classes = tf.reshape(logits, [-1, 5])

    labels_flat = tf.reshape(label_batch, [-1,])

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits_by_num_classes, labels=labels_flat)
    loss = cross_entropy





    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9)

    train_op = optimizer.minimize(loss, global_step)

    sess = tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(3000):
        print(i)
        keys = random.sample(label.keys(),5)
        feature_batch = np.array([feature[key] for key in keys])
        feature_batch_shape = feature_batch.shape
        feature_batch = feature_batch.reshape(feature_batch_shape + (1,))
        y_batch = np.array([label[key] for key in keys])
        sess.run(train_op, feed_dict = {input_batch: feature_batch,is_training: True, label_batch:y_batch})
        if i % 10 == 0:
            print("eval:")
            keys_test = list(label.keys())[:4]
            feature_test = np.array([feature[key] for key in keys])
            feature_test_shape = feature_test.shape
            feature_test = feature_test.reshape(feature_test_shape + (1,))
            y_test = np.array([label[key] for key in keys])            
            print(sess.run(cross_entropy, feed_dict = {input_batch: feature_batch,is_training: True, label_batch:y_batch}))

writer = tf.summary.FileWriter(r'Z:\Kaggle\Steel\graph')
writer.add_graph(sess.graph)
