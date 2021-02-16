# Copyright 2021 by Andrey Ignatov. All Rights Reserved.
#
# IMPORTANT NOTE:
#
# Since the min distance to the object is equal to 0.4m = 400mm, we can replace all unknown values (marked by 0s) with ones.
# In this case, there won't be any problems when computing logarithms in the formulas below (since log[x >= 1] >= 0).

import numpy as np
np.random.seed(42)


def rmse(img, target):

    # If the input tensors are not floating-point ones, first convert them to the corresponding format:
    # img = np.asarray(img, dtype=np.float)
    # target = np.asarray(target, dtype=np.float)

    img[img < 1] = 1
    target[target < 1] = 1
    mask = np.asarray(target > 1, dtype=int)

    diff = (target - img) * mask / 1000.0   # mapping the distance from millimeters to meters
    num_pixels = float(np.sum(mask > 0))

    return np.sqrt(np.sum(np.square(diff)) / num_pixels)


def si_rmse(img, target):

    # If the input tensors are not floating-point ones, first convert them to the corresponding format:
    # img = np.asarray(img, dtype=np.float)
    # target = np.asarray(target, dtype=np.float)

    img[img < 1] = 1
    target[target < 1] = 1
    mask = np.asarray(target > 1, dtype=int)

    log_diff = (np.log(img) - np.log(target)) * mask
    num_pixels = float(np.sum(mask > 0))

    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


def avg_log10(img, target):

    # If the input tensors are not floating-point ones, first convert them to the corresponding format:
    # img = np.asarray(img, dtype=np.float)
    # target = np.asarray(target, dtype=np.float)

    img[img < 1] = 1
    target[target < 1] = 1
    mask = np.asarray(target > 1, dtype=int)

    log_diff = (np.log10(img) - np.log10(target)) * mask
    num_pixels = float(np.sum(mask > 0))

    return np.sum(np.absolute(log_diff)) / num_pixels


def rel(img, target):

    # If the input tensors are not floating-point ones, first convert them to the corresponding format:
    # img = np.asarray(img, dtype=np.float)
    # target = np.asarray(target, dtype=np.float)

    img[img < 1] = 1
    target[target < 1] = 1
    mask = np.asarray(target > 1, dtype=int)

    diff = (img - target) * mask
    num_pixels = float(np.sum(mask > 0))

    return np.sum(np.absolute(diff) / target) / num_pixels


if __name__=='__main__':

    x = np.random.randint(0, 10000, (480, 640))
    y = np.random.randint(0, 10000, (480, 640))

    print(rmse(x, y))
    print(si_rmse(x, y))
    print(avg_log10(x, y))
    print(rel(x, y))
