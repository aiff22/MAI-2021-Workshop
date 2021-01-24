import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import imageio


def representative_dataset():

    dataset_size = 10

    for i in range(dataset_size):

        data = imageio.imread("sample_images/" + str(i) + ".jpg")
        data = np.reshape(data, [1, 384, 576, 3])
        yield [data.astype(np.float32)]


def convert_model():

    # Define the input layer of your network and initialize the MobileNetV2 model

    input_size = (384, 576, 3)

    input_image = layers.Input(shape=input_size)
    input_image_normalized = preprocess_input(input_image)

    sample_model = MobileNetV2(input_tensor=input_image_normalized, input_shape=input_size, include_top=False)

    for layer in sample_model.layers:
        layer.trainable = False

    # Get the features obtained after the last layer of the model
    features_layer = sample_model.get_layer('out_relu')
    features_output = features_layer.output

    x = layers.Flatten()(features_output)
    x = layers.Dense(64, activation='sigmoid')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(30, activation='softmax')(x)

    # Define the entire model
    model = Model(input_image, output)

    # Load your pre-trained model
    # model.load_weights("path/to/your/saved/model")

    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

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
