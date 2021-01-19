# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to convert a simple Keras U-Net based model to TFLite format

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.optimizers import Adam


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

    conv_last = layers.Conv2D(3, kernel_size=3, padding='same', activation=None)(conv9)

    model = Model(inputs=[image], outputs=[conv_last])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])

    return model


def convert_model():

    # Defining the model
    model = UNet((720, 1280, 3))

    # Load your pre-trained model
    # model.load_weights("path/to/your/saved/model")

    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

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


convert_model()
