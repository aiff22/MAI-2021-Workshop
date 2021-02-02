import h5py
import numpy as np
import tensorflow as tf
import cv2

class TrainDataset(tf.keras.utils.Sequence):
    def __init__(self, h5_file, batch_size=32):
        self.h5_file = h5_file
        self.batch_size = batch_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx * self.batch_size:(idx + 1) * self.batch_size] / 255., -1), np.expand_dims(f['hr'][idx * self.batch_size:(idx + 1) * self.batch_size] / 255., -1)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr']) // self.batch_size

def representative_dataset_gen():
    with h5py.File('datasets/div2k_x4_sample.h5', 'r') as f:
        for lr in f['lr']:
            yuv = cv2.cvtColor(lr, cv2.COLOR_BGR2YUV)
            y, _, _ = cv2.split(yuv)
            y = y.astype(np.float32) / 255.
            y = np.expand_dims(y, 0)
            y = np.expand_dims(y, -1)
            yield [y]

def representative_dataset_gen_image_shapes():
    with h5py.File('datasets/div2k_x4_sample.h5', 'r') as f:
        lr = f['lr'][0]
        hr = f['hr'][0]
        return [1, lr.shape[0], lr.shape[1], 1], [1, hr.shape[0], hr.shape[1], 1]
