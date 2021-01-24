# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to test your converted (quantized / floating-point) TFLite model
# on the real images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import os

from tensorflow.lite.python import interpreter as interpreter_wrapper


if __name__ == "__main__":

    # Specify the name of your TFLite model and the location of the sample test images

    image_folder = "sample_images/"
    model_file = "model.tflite"

    # Load your TFLite model

    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = False

    if input_details[0]['dtype'] == type(np.float32(1.0)):
        floating_model = True

    # Get the size of the input / output tensors

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Process test images and display the results

    image_list = os.listdir(image_folder)

    for image_name in image_list:

        img = Image.open(image_folder + image_name)
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = np.float32(input_data)
        else:
            input_data = np.uint8(input_data)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        prediction = np.argmax(results)
        prediction_top_3 = results.argsort()[-3:][::-1]

        print("Image " + image_name + ", predicted classes: \t %d, %d, %d" %
              (prediction_top_3[0], prediction_top_3[1], prediction_top_3[2]))
