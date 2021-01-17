# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to convert a simple TensorFlow U-Net based model to TFLite format

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


def UNet(image, base_filters=16):

    s = base_filters

    conv1, pool1 = double_conv(image, [s, s])
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

    return tf.compat.v1.layers.conv2d(conv9, 3, (1, 1), name='conv_final', activation=None, padding='same')


with tf.compat.v1.Session() as sess:

    # Placeholders for input data
    # The values of the input image should lie in the interval [0, 255]
    # ------------------------------------------------------------------
    x_ = tf.compat.v1.placeholder(tf.float32, [1, 720, 1280, 3], name="input")

    # Perform image preprocessing if needed (e.g., normalization, scaling, etc.)
    x_normalized = x_ / 255.0

    # Process the image with a sample SRCNN model
    processed = UNet(x_normalized)

    # Scale the processed image so that its values lie in the interval [0, 255]
    output_ = tf.identity(processed * 255, name="output")

    # Load your pre-trained model
    # saver = tf.compat.v1.train.Saver()
    # saver.restore(sess, "path/to/your/saved/model")

    # In this example, we will just initialize the model with some random values
    sess.run(tf.compat.v1.global_variables_initializer())

    # Export your model to the TFLite format
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [x_], [output_])

    # Be very careful here:
    # "experimental_new_converter" is enabled by default in TensorFlow 2.2+. However, using the new MLIR TFLite
    # converter might result in corrupted / incorrect TFLite models for some particular architectures. Therefore, the
    # best option is to perform the conversion using both the new and old converter and check the results in each case:
    converter.experimental_new_converter = False

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------
