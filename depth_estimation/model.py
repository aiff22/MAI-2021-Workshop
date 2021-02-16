# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf


def double_conv(net, filters, pool=True):

    for filter_size in filters:
        net = tf.compat.v1.layers.conv2d(net, filter_size, (3, 3), activation="relu", padding='same')

    if pool:
        return net, tf.compat.v1.layers.max_pooling2d(net, (2, 2), strides=(2, 2))
    else:
        return net


def upconv_concat(net, tensor_concat, filters):

    net = tf.compat.v1.layers.conv2d_transpose(net, filters=filters, kernel_size=2, strides=2)
    return tf.concat([net, tensor_concat], axis=-1)


def UNet(image, base_filters=64):

    s = base_filters

    with tf.compat.v1.variable_scope("model"):

        downscaled_2x = tf.compat.v1.image.resize_bilinear(image, (240, 320), half_pixel_centers=False, align_corners=False)

        conv1, pool1 = double_conv(downscaled_2x, [s, s])
        conv2, pool2 = double_conv(pool1, [s*2, s*2])
        conv3, pool3 = double_conv(pool2, [s*4, s*4])
        conv4, pool4 = double_conv(pool3, [s*8, s*8])
        conv5 = double_conv(pool4, [s*16, s*16], pool=False)

        up6 = upconv_concat(conv5, conv4, s*8)
        conv6 = double_conv(up6, [s*8, s*8], pool=False)

        up7 = upconv_concat(conv6, conv3, s*4)
        conv7 = double_conv(up7, [s*4, s*4], pool=False)

        up8 = upconv_concat(conv7, conv2, s*2)
        conv8 = double_conv(up8, [s*2, s*2], pool=False)

        up9 = upconv_concat(conv8, conv1, s)
        conv9 = double_conv(up9, [s, s], pool=False)

        conv_final = tf.compat.v1.layers.conv2d(conv9, 1, (3, 3), name='conv_final', activation="relu", padding='same')

        return tf.compat.v1.image.resize_bilinear(conv_final, (480, 640), half_pixel_centers=False, align_corners=False)

