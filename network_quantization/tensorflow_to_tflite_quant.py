# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to get Android NNAPI-compatible quantized TFLite U-Net model
#
# Important note:
#
# The following code works with TensorFlow 2.4.0. Due to a long-lasting tradition of breaking all functionality
# and changing the API with each TF / TFLite build, the resulting models obtained with different TensorFlow versions
# will also be different and might sometimes be corrupted, though the code is the same.

import tensorflow as tf
import numpy as np
import imageio


def double_conv(net, filters, pool=True):

    for filter_size in filters:
        net = tf.compat.v1.layers.conv2d(net, filter_size, (3, 3), activation="relu", padding='same')

    if pool:
        return net, tf.compat.v1.layers.max_pooling2d(net, (2, 2), strides=(2, 2))
    else:
        return net


def upconv_concat(net, tensor_concat, filters, kernel_size=2):

    net = tf.compat.v1.layers.conv2d_transpose(net, filters=filters, kernel_size=kernel_size, strides=2)
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

    up10 = tf.compat.v1.layers.conv2d_transpose(conv9, filters=s, kernel_size=3, strides=3)
    return tf.compat.v1.layers.conv2d(up10, 3, (1, 1), name='conv_final', activation=None, padding='same')


def representative_dataset():

    dataset_size = 10

    for i in range(dataset_size):
        print(i)
        data = imageio.imread("sample_images/" + str(i) + ".jpg")
        data = np.reshape(data, [1, 480, 720, 3])
        yield [data.astype(np.float32)]


with tf.compat.v1.Session() as sess:

    # Placeholders for input data
    # The values of the input image should lie in the interval [0, 255]
    # ------------------------------------------------------------------
    image = tf.compat.v1.placeholder(tf.float32, [1, 480, 720, 3], name="input")

    # Process the image with a sample SRCNN model
    processed = UNet(image)

    output = tf.identity(processed, name="output")

    # Load your pre-trained model
    # saver = tf.compat.v1.train.Saver()
    # saver.restore(sess, "path/to/your/saved/model")

    # In this example, we will just initialize the model with some random values
    sess.run(tf.compat.v1.global_variables_initializer())

    # Export your model to the TFLite format
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [image], [output])

    # Be very careful here:
    # "experimental_new_converter" is enabled by default in TensorFlow 2.2+. However, using the new MLIR TFLite
    # converter might result in corrupted / incorrect TFLite models for some particular architectures. Therefore, the
    # best option is to perform the conversion using both the new and old converter and check the results in each case:
    converter.experimental_new_converter = False
    converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # If you are performing input data normalization (e.g., scaling or zero centering), it is better to remove
    # the corresponding codes from your final pre-trained model and do this normalization using TFLite:
    #
    # Specify input normalization:
    #
    # No normalization, (0, 255) range:             {input_arrays[0]: (0.0, 1.0)}
    # Scaling only, (0, 1) range:                   {input_arrays[0]: (0.0, 255.0)}
    # Zero centering + scaling, (-1, 1) range:      {input_arrays[0]: (127.5, 127.5)}
    #
    # More information can be found here:
    # https://stackoverflow.com/questions/54830869/understanding-tf-contrib-lite-tfliteconverter-quantization-parameters
    #
    # Note that this norm options should be used INSTEAD of input data normalization in your code, but NOT TOGETHER.
    # I.e., you are training your model with your own norm implementation, then removing it when restoring the model
    # and adding the following options instead:

    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------
