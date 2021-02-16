# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf


def rmse(img, target, mask, num_pixels):

    diff = tf.math.multiply(img - target, mask) / 1000.0    # mapping the distance from meters to millimeters

    loss_mse = tf.reduce_sum(tf.pow(diff, 2)) / num_pixels
    loss_rmse = tf.sqrt(loss_mse)

    return loss_rmse


def si_rmse(img, target, mask, num_pixels):

    log_diff = tf.math.multiply((tf.math.log(img) - tf.math.log(target)), mask)

    loss_si_rmse = tf.sqrt(tf.reduce_sum(tf.square(log_diff)) / num_pixels -
                           tf.square(tf.reduce_sum(log_diff)) / tf.square(num_pixels))

    return loss_si_rmse


def avg_log10(img, target, mask, num_pixels):

    log_diff_10 = tf.math.multiply(((tf.math.log(img) - tf.math.log(target)) / tf.math.log(tf.constant(10.0))), mask)

    loss_log10 = tf.reduce_sum(tf.abs(log_diff_10)) / num_pixels

    return loss_log10


def rel(img, target, mask, num_pixels):

    diff = tf.math.multiply((img - target), mask)

    loss_rel = tf.reduce_sum(tf.math.divide(tf.abs(diff), target)) / num_pixels

    return loss_rel

