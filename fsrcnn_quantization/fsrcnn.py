import sys
import tensorflow as tf
import h5py
import math
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, PReLU, ReLU
from tensorflow.keras.models import Model
from dataset import TrainDataset, representative_dataset_gen, representative_dataset_gen_image_shapes


def build(scale_factor=4, num_channels=1, d=56, s=12, m=4):
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


def train(model, dataset_path, num_epochs=1, batch_size=32):
    train_gen = TrainDataset(dataset_path, batch_size=batch_size)
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_gen, epochs=num_epochs, workers=8)
    return model

def quantize(saved_model_path):
    # Load trained SavedModel
    model = tf.saved_model.load(saved_model_path)
    # Setup fixed input shape
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    lr_shape, hr_shape = representative_dataset_gen_image_shapes()
    concrete_func.inputs[0].set_shape(lr_shape)
    # Get tf.lite converter instance
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Use full integer operations in quantized model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set input and output dtypes to UINT8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # Provide representative dataset for quantization calibration
    converter.representative_dataset = representative_dataset_gen
    # Convert to 8-bit TensorFlow Lite model
    return converter.convert()

def evaluate(model_file, image_index=0, test_dataset='datasets/div2k_x4_sample.h5'):

    def calc_psnr(y, y_target):
        mse = np.mean((y - y_target) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10( 255. / math.sqrt(mse))

    with h5py.File(test_dataset, 'r') as f:
        lr = f['lr'][image_index]
        hr = f['hr'][image_index]

    yuv = cv2.cvtColor(lr, cv2.COLOR_BGR2YUV)
    lr_y, _, _ = cv2.split(yuv)
    lr_y = np.expand_dims(lr_y, 0)
    lr_y = np.expand_dims(lr_y, -1)

    if model_file[-7:] == '.tflite':
        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], lr_y)
        interpreter.invoke()
        sr_y = interpreter.get_tensor(output_details[0]['index']).squeeze()
        sr_y = sr_y.astype(np.float32)
        mean = output_details[0]['quantization'][1]
        scale = output_details[0]['quantization'][0]
        if scale == 0:
            scale = 1
        sr_y = (sr_y - mean ) * scale * 255.0
    else:
        print('Unsupported model file')
        raise ValueError

    sr_y = np.clip(np.round(sr_y), 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(hr, cv2.COLOR_BGR2YUV)
    hr_y, _, _ = cv2.split(yuv)
    return sr_y, calc_psnr(sr_y, hr_y)
