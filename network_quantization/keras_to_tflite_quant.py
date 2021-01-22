# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to get Android NNAPI-compatible quantized TFLite U-Net model
#
# Important note:
#
# The following code works with TensorFlow 2.4.0. Due to a long-lasting tradition of breaking all functionality
# and changing the API with each TF / TFLite build, the resulting models obtained with different TensorFlow versions
# will also be different and might sometimes be corrupted, though the code is the same.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.optimizers import Adam

import imageio
import numpy as np


def double_conv(net, filters, pool=True):

    for filter_size in filters:
        net = layers.Conv2D(filter_size, kernel_size=3, activation="relu", padding='same')(net)

    if pool:
        return net, layers.MaxPool2D(pool_size=(2, 2))(net)
    else:
        return net


def upconv_concat(net, tensor_concat):

    net = layers.UpSampling2D(size=(2, 2))(net)
    return layers.Concatenate(axis=-1)([net, tensor_concat])


def UNet(input_size, base_filters=16):

    s = base_filters
    image = keras.Input(shape=input_size)

    conv1, pool1 = double_conv(image, [s, s])
    conv2, pool2 = double_conv(pool1, [s*2, s*2])
    conv3, pool3 = double_conv(pool2, [s*4, s*4])
    conv4, pool4 = double_conv(pool3, [s*8, s*8])
    conv5 = double_conv(pool4, [s*16, s*16], pool=False)

    up6 = upconv_concat(conv5, conv4)
    conv6 = double_conv(up6, [s * 8, s * 8], pool=False)

    up7 = upconv_concat(conv6, conv3)
    conv7 = double_conv(up7, [s * 4, s * 4], pool=False)

    up8 = upconv_concat(conv7, conv2)
    conv8 = double_conv(up8, [s * 2, s * 2], pool=False)

    up9 = upconv_concat(conv8, conv1)
    conv9 = double_conv(up9, [s, s], pool=False)

    up10 = layers.UpSampling2D(size=(3, 3))(conv9)
    conv_last = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(up10)

    model = Model(inputs=[image], outputs=[conv_last])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])

    return model


def representative_dataset():

    dataset_size = 10

    for i in range(dataset_size):
        print(i)
        data = imageio.imread("sample_images/" + str(i) + ".jpg")
        data = np.reshape(data, [1, 480, 720, 3])
        yield [data.astype(np.float32)]


def convert_model():

    # Defining the model
    model = UNet((480, 720, 3))

    # Load your pre-trained model
    # model.load_weights("path/to/your/saved/model")

    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Be very careful here:
    # "experimental_new_converter" is enabled by default in TensorFlow 2.2+. However, using the new MLIR TFLite
    # converter might result in corrupted / incorrect TFLite models for some particular architectures. Therefore, the
    # best option is to perform the conversion using both the new and old converter and check the results in each case:
    converter.experimental_new_converter = False
    converter._experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------


convert_model()
