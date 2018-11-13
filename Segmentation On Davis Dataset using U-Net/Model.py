import tensorflow as tf
from tensorflow.contrib.keras import layers
import numpy as np

def crop_and_concat(x1,x2):

    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)

    x2_shape1 = x2.get_shape()
    x1_shape1 = x1.get_shape()

    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    print("offsets",offsets)
    print("x1",x1)

    #offsets = [0, (int(x1_shape1[1]) - int(x2_shape1[1])) // 2, (int(x1_shape1[2]) - int(x2_shape1[2])) // 2, 0]
    size = [-1, int(x2_shape1[1]), int(x2_shape1[2]), -1]
    print("size",size)
    #size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    print("x1 crop",x1_crop)
    return tf.concat([x1_crop, x2], 3)

def build_model(x, keep_prob, batch_size):
    print(x.get_shape())
    conv1 = tf.layers.conv2d(
        inputs= x,
        filters=32,
        kernel_size=(3,3),
        padding='same',
        activation = tf.nn.relu
    )#256
    print("conv1", conv1)
    conv1_2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=(3,3),
        padding='same',
        activation=tf.nn.relu
    )#252
    print("conv1", conv1_2)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_2,
        pool_size=(2,2),
        padding='same',
        strides = 2
    )#126
    print("pool1", pool1)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )#124
    print("conv2", conv2.get_shape())
    conv2 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )#122
    print("conv2", conv2.get_shape())
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2
    )#61
    print("pool2", pool2.get_shape())
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )#59
    print("conv3", conv3.get_shape())
    conv3 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )#57
    print("conv3", conv3.get_shape())
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=2,
        strides=2
    )
    print("pool3", pool3.get_shape())
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv4", conv4.get_shape())
    conv4 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )
    print("conv4", conv4.get_shape())
    pool4 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=(2,2),
        strides=2
    )
    print("pool4", pool4.get_shape())

    conv4_1 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv41", conv4_1.get_shape())
    conv4_1 = tf.layers.conv2d(
        inputs=conv4_1,
        filters=512,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )
    print("conv41", conv4_1.get_shape())
    pool4_1 = tf.layers.max_pooling2d(
        inputs=conv4_1,
        pool_size=(2, 2),
        strides=2
    )
    print("pool41", pool4_1.get_shape())
    conv5 = tf.layers.conv2d(
        inputs=pool4_1,
        filters=1024,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv5", conv5.get_shape())
    conv5 = tf.layers.conv2d(
        inputs=conv5,
        filters=1024,
        kernel_size=(3,3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv5", conv5.get_shape())

    up1_0 = layers.concatenate([layers.UpSampling2D(size=(2,2))(conv5),conv4_1])
    #up1_0 = crop_and_concat(layers.UpSampling2D(size=(2,2))(conv5), conv4_1)

    print("up10", up1_0.get_shape())
    conv6 = tf.layers.conv2d(
        inputs=up1_0,
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv6", conv6.get_shape())
    conv6 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv6", conv6.get_shape())
    up1 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv6), conv4])
    #up1 = crop_and_concat(layers.UpSampling2D(size=(2,2))(conv6), conv4)
    print("up1", up1.get_shape())
    conv6_1 = tf.layers.conv2d(
        inputs=up1,
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv61", conv6_1.get_shape())
    conv6_1 = tf.layers.conv2d(
        inputs=conv6_1,
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv61", conv6_1.get_shape())
    up2  = layers.concatenate([layers.UpSampling2D(size=2)(conv6_1),conv3])
    #up2 = crop_and_concat(layers.UpSampling2D(size=(2,2))(conv6_1), conv3)
    print("up2", up2.get_shape())
    conv7 = tf.layers.conv2d(
        inputs=up2,
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv7", conv7.get_shape())
    conv7 = tf.layers.conv2d(
        inputs=conv7,
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv7", conv7.get_shape())
    up3 =  layers.concatenate([layers.UpSampling2D(size=2)(conv7),conv2])
    #up3 = crop_and_concat(layers.UpSampling2D(size=(2,2))(conv7), conv2)
    print("up3", up3.get_shape())
    conv8 = tf.layers.conv2d(
        inputs=up3,
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv8", conv8.get_shape())
    conv8 = tf.layers.conv2d(
        inputs=conv8,
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv8",conv8.get_shape())
    up4 = layers.concatenate([layers.UpSampling2D(size=2)(conv8), conv1])
    #up4 = crop_and_concat(layers.UpSampling2D(size=(4,4))(conv8), conv1)
    print("up4",up4.get_shape())
    conv9 = tf.layers.conv2d(
        inputs=up4,
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv91",conv9.get_shape())
    conv9 = tf.layers.conv2d(
        inputs=conv9,
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation=tf.nn.relu
    )
    print("conv9",conv9.get_shape())
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=1,
        kernel_size=1,
        activation= tf.nn.sigmoid
    )
    print("conv10",conv10.get_shape())
    return conv10


