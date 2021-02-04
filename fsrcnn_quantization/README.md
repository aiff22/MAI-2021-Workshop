# FSRCNN Super Resolution Quantization Demo

Demonstrates training and quantization of single image super resolution using a Keras implementation of FSRCNN (http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

The demo has been prepared to train, quantize, and evaluate FSRCNN with an upscaling factor of 3 and the DIV2K dataset. For other scale factors, training and quantization data of the correct sizes will need to be provided by the user.

Quantization utilizes TensorFlow's post training integer quantization tools (https://www.tensorflow.org/lite/performance/post_training_integer_quant).

Input shape for the quantized TensorFlow Lite model is [1, 360, 640, 1], output shape is [1, 1080, 1920, 1].

The DIV2K dataset should be organized as follows:
```
DIV2K   
├── DIV2K_train_HR  
└── DIV2K_train_LR_bicubic  
```
1. Python3 pip Requurements  
h5py  
numpy  
opencv-python  
tensorflow==2.3.1  

2. Train  
$ python3 demo.py --train --div2k-root /path/to/DIV2K  
==> fsrcnn_saved_model/ (trained SavedModel)  

3. Quantize  
$ python3 demo.py --quantize --saved-model-path fsrcnn_saved_model --div2k-root /path/to/DIV2K  
==> ./fsrcnn_x3_ptq.tflite (8-bit TensorFlow Lite model)  

4. Evaluate  
$ python3 demo.py --eval --quant-model-path fsrcnn_x3_ptq.tflite --div2k-root /path/to/DIV2K --eval-index 0 (index range [0-800])  
==> ./sr_y_0.png (upscaled Y-channel evaluation image)  
