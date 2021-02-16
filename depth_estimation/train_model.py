# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np
import imageio
import os

from model import UNet
from losses_tf import *

np.random.seed(42)

IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_CHANNELS = 640, 480, 1

# Modify the model training parameters below:

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_ITERATIONS = 100000
EVAL_STEP = 1000

TRAIN_DIR = "challenge_data/train/"
VAL_DIR = "challenge_data/validation/"

NUM_TRAIN_IMAGES = 1600
NUM_VAL_IMAGES = len(os.listdir(VAL_DIR + "rgb/"))
NUM_VAL_BATCHES = int(NUM_VAL_IMAGES / BATCH_SIZE)


def load_data(dir_name, num_images=-1):

    rgb_image_dir = dir_name + 'rgb/'
    depth_image_dir = dir_name + 'depth/'

    image_list = os.listdir(rgb_image_dir)
    dataset_size = len(image_list)

    image_ids = np.random.choice(np.arange(0, dataset_size), num_images, replace=False)

    rgb_data = np.zeros((num_images, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    depth_data = np.zeros((num_images, IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS))

    i = 0
    for img_id in image_ids:

        I_rgb = imageio.imread(rgb_image_dir + image_list[img_id])
        I_depth = imageio.imread(depth_image_dir + image_list[img_id])

        rgb_data[i] = np.reshape(I_rgb, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        depth_data[i] = np.reshape(I_depth, [IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS])

        i += 1

    # Setting the lower bound to 1mm to avoid problems when computing logarithms
    depth_data[depth_data < 1] = 1

    return rgb_data, depth_data


with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:

    # Placeholders for training data

    input_ = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    target_ = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS])
    mask_ = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS])
    num_pixels_ = tf.compat.v1.placeholder(tf.float32)

    input_norm = input_ / 255.0         # mapping to [0, 1] interval

    # Get the predicted depth distance (in meters) and map it to millimeters

    predictions_raw = UNet(input_norm) * 1000

    # Clip the obtained values to uint16. The lower bound is set to 1mm to avoid problems when computing logarithms

    predictions = tf.clip_by_value(predictions_raw, 1.0, 65535.0)
    final_outputs = tf.cast(tf.clip_by_value(predictions_raw, 0.0, 65535.0), tf.uint16)

    # Losses

    loss_rmse = rmse(predictions, target_, mask_, num_pixels_)
    loss_si_rmse = si_rmse(predictions, target_, mask_, num_pixels_)
    loss_log10 = avg_log10(predictions, target_, mask_, num_pixels_)
    loss_rel = rel(predictions, target_, mask_, num_pixels_)

    # Will be training the model with a simple RMSE loss function
    loss_total = loss_rmse

    # Optimize network parameters

    train_step = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss_total)
    model_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("model")]

    # Initialize and restore the variables

    print("Initializing variables")

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(var_list=model_vars, max_to_keep=100)

    print("Loading training data...")
    train_data, train_targets = load_data(TRAIN_DIR, NUM_TRAIN_IMAGES)
    print("Training data was loaded\n")

    print("Loading validation data...")
    val_data, val_targets = load_data(VAL_DIR, NUM_VAL_IMAGES)
    print("Test data was loaded\n")

    # Select a couple of val images for visualizing the results
    visual_image_ids = np.random.choice(np.arange(0, NUM_VAL_IMAGES), BATCH_SIZE, replace=False)
    visual_test_images = val_data[visual_image_ids, :]
    visual_target_targets = val_targets[visual_image_ids, :]

    print("Training network")

    for i in range(NUM_TRAIN_ITERATIONS):

        # Train model

        idx_train = np.random.choice(np.arange(0, NUM_TRAIN_IMAGES), BATCH_SIZE, replace=False)

        input_images = train_data[idx_train]
        target_images = train_targets[idx_train]

        target_mask = np.asarray(target_images > 1, dtype=np.float)
        target_mask = np.reshape(target_mask, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS])
        num_pixels = float(np.sum(target_mask))

        sess.run(train_step, feed_dict={input_: input_images, target_: target_images,
                                        mask_: target_mask, num_pixels_: num_pixels})

        # Evaluate model

        if i % EVAL_STEP == 0:

            loss_rmse_eval = 0.0
            loss_si_rmse_eval = 0.0
            loss_log10_eval = 0.0
            loss_rel_eval = 0.0

            for j in range(NUM_VAL_BATCHES):

                be = j * BATCH_SIZE
                en = (j + 1) * BATCH_SIZE

                input_eval = val_data[be:en]
                target_eval = val_targets[be:en]

                target_mask = np.asarray(target_eval > 1, dtype=np.float)
                target_mask = np.reshape(target_mask, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS])
                num_pixels = float(np.sum(target_mask))

                losses = sess.run([loss_rmse, loss_si_rmse, loss_log10, loss_rel],
                                  feed_dict={input_: input_eval, target_: target_eval, mask_: target_mask, num_pixels_: num_pixels})

                loss_rmse_eval += losses[0] / NUM_VAL_BATCHES
                loss_si_rmse_eval += losses[1] / NUM_VAL_BATCHES
                loss_log10_eval += losses[2] / NUM_VAL_BATCHES
                loss_rel_eval += losses[3] / NUM_VAL_BATCHES

            logs_losses = "step %d | RMSE: %.4g, SI_RMSE: %.4g, LOG_10: %.4g, REL: %.4g" % \
                          (i, loss_rmse_eval, loss_si_rmse_eval, loss_log10_eval, loss_rel_eval)
            print(logs_losses)

            # Save the model that corresponds to the current iteration
            saver.save(sess, "models/unet_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

            # Save visual results for several test images
            visual_results = sess.run(final_outputs, feed_dict={input_: visual_test_images})

            idx = 0
            for image in visual_results:
                if idx < 4:

                    input_image = np.asarray(np.reshape(visual_test_images[idx], [IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=np.uint8)
                    target_image = np.asarray(np.reshape(visual_target_targets[idx], [IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS]), dtype=np.uint16)
                    predicted_image = np.asarray(np.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS]), dtype=np.uint16)

                    imageio.imsave("results/" + str(idx) + ".jpg", input_image)
                    imageio.imsave("results/" + str(idx) + "_depth.png", target_image)
                    imageio.imsave("results/" + str(idx) + "_iter_" + str(i) + ".png", predicted_image)

                idx += 1

        # Loading new training data

        if i % 1000 == 0:

            del train_data
            del train_targets
            train_data, train_targets = load_data(TRAIN_DIR, NUM_TRAIN_IMAGES)
