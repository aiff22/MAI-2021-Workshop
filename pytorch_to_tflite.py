# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# This code was checked to work with the following library versions:
#
# ONNX-TensorFlow:  1.7.0   [pip install onnx-tf==1.7.0]
# ONNX:             1.8.0   [pip install onnx==1.8.0]
# TensorFlow:       2.4.0   [pip install tensorflow==2.4.0]
# PyTorch:          1.7.1   [pip install ]
#
# More information about ONNX-TensorFlow: https://github.com/onnx/onnx-tensorflow

import torch.nn as nn
import torch
import os

# DO NOT COMMENT THIS LINE (IT IS DISABLING GPU)!
# WHEN COMMENTED, THE RESULTING TF MODEL WILL HAVE INCORRECT LAYER FORMAT
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import onnx
from onnx_tf.backend import prepare

import tensorflow as tf


def conv_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
    )


def conv_conv_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2), nn.ReLU()
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down_1 = conv_conv_2(3, 16)
        self.down_2 = conv_conv_2(16, 32)
        self.down_3 = conv_conv_2(32, 64)
        self.down_4 = conv_conv_2(64, 128)

        self.bottom = conv_conv(128, 128)

        self.up_1 = conv_conv(128, 64)
        self.up_2 = conv_conv(64, 32)
        self.up_3 = conv_conv(32, 16)

        self.conv_final = nn.Conv2d(16, 3, 1, padding=0)

        self.upsample_0 = torch.nn.Upsample(scale_factor=2)
        self.upsample_1 = torch.nn.Upsample(scale_factor=2)
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)
        self.upsample_3 = torch.nn.Upsample(scale_factor=2)

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)

        x = self.upsample_0(self.bottom(x))
        x = self.upsample_1(self.up_1(x))
        x = self.upsample_2(self.up_2(x))
        x = self.upsample_3(self.up_3(x))

        return self.conv_final(x)


if __name__ == '__main__':

    # Creating / loading pre-trained PyNET model

    model = UNet()
    model.eval()

    # Converting model to ONNX

    for _ in model.modules():
        _.training = False

    sample_input = torch.randn(1, 3, 720, 1280)
    input_nodes = ['input']
    output_nodes = ['output']

    torch.onnx.export(model, sample_input, "model.onnx", export_params=True, input_names=input_nodes, output_names=output_nodes)

    # Converting model to Tensorflow

    onnx_model = onnx.load("model.onnx")
    output = prepare(onnx_model)
    output.export_graph("tf_model/")

    # Exporting the resulting model to TFLite

    converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
