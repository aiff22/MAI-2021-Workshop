import sys
import tensorflow as tf
import h5py
import math
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, PReLU, ReLU
from tensorflow.keras.models import Model
from dataset import DIV2K


def build(scale_factor=3, num_channels=1, d=56, s=12, m=4):
    """Implements FSRCNN in Keras
    http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
    """
    inp = Input(shape=(None, None, 1))
    # Feature extraction
    x = Conv2D(d, 5, padding='same')(inp)
    x = ReLU()(x)
    # Shrinking
    x = Conv2D(s, 1, padding='valid')(x)
    x = ReLU()(x)
    # Mapping
    for _ in range(m):
        x = Conv2D(s, 3, padding='same')(x)
        x = ReLU()(x)
    # Expanding
    x = Conv2D(d, 1, padding='same')(x)
    x = ReLU()(x)
    # Deconvolution
    out = Conv2DTranspose(num_channels, 9, strides=scale_factor, padding='same')(x)
    return Model(inputs=inp, outputs=out)


def train(model, dataset_path, scale_factor=3, num_epochs=1, batch_size=32):
    train_gen = DIV2K(dataset_path, scale_factor=scale_factor, batch_size=batch_size)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_gen, epochs=num_epochs, workers=8)
    return model

def quantize(saved_model_path, data_path, input_shape, scale_factor):
    def representative_dataset_gen():
        div2k = DIV2K(data_path, scale_factor=scale_factor, patch_size=0)
        for i in range(50):
            x, _ = div2k[i]
            # Skip images that are not witin input h,w boundaries
            if x.shape[0] > input_shape[1] and x.shape[1] > input_shape[2]:
                # crop to input shape starting for top left corner of image
                x = x[:input_shape[1], :input_shape[2]]
                x = np.expand_dims(x, 0)
                x = np.expand_dims(x, -1)
                yield [x]
    # Load trained SavedModel
    model = tf.saved_model.load(saved_model_path)
    # Setup fixed input shape
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(input_shape)
    # Get tf.lite converter instance
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Use full integer operations in quantized model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set input and output dtypes to UINT8 (uncomment the following two lines to generate an integer only model)
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # Provide representative dataset for quantization calibration
    converter.representative_dataset = representative_dataset_gen
    # Convert to 8-bit TensorFlow Lite model
    return converter.convert()

def evaluate(model_file, data_path, image_index=0):

    def calc_psnr(y, y_target):
        mse = np.mean((y - y_target) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10( 1. / math.sqrt(mse))

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']
    div2k = DIV2K(data_path, patch_size=0)
    # Get lr, hr image pair
    lr, hr = div2k[image_index]
    # Check if image size can be used for inference
    if lr.shape[0] < input_shape[1] or lr.shape[1] < input_shape[2]:
        print(f'Eval image {image_index} has invalid dimensions. Expecting h >= {input_shape[1]} and w >= {input_shape[2]}.')
        raise ValueError
    # Crop lr, hr images to match fixed shapes of the tensorflow lite model
    lr = lr[:input_shape[1], :input_shape[2]]
    lr = np.expand_dims(lr, 0)
    lr = np.expand_dims(lr, -1)
    interpreter.set_tensor(input_details[0]['index'], lr)
    interpreter.invoke()
    sr = interpreter.get_tensor(output_details[0]['index']).squeeze()
    hr = hr[:output_shape[1], :output_shape[2]]
    return np.clip(np.round(sr * 255.), 0, 255).astype(np.uint8), np.clip(np.round(hr * 255.), 0, 255).astype(np.uint8), calc_psnr(sr, hr)
