import cv2
import numpy as np
import os

def display(name, image):
    """displays an image for ten seconds or until key is pressed.

    NOTE: exiting out of the image blocks the program until the ten 
    seconds elapse. It is suggested you press a key instead of closing
    the window"""
    cv2.imshow(name, image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def filter(image, kernel, scale = 1.0):
    """applies a filter to the image"""
    mat = np.array(kernel) * scale
    image = cv2.filter2D(image, -1, mat)
    return image

def gaussian(stddev, size = 3):
    """calculates a centered gaussian kernel"""
    ret = np.ndarray((size, size))
    gauss = lambda x, y : np.exp(-(x*x + y*y) / (2 * np.square(stddev)))
    for i in range(size):
        for j in range(size):
            ret[i,j] = gauss(i - (size-1) / 2, j - (size-1) / 2)
    return ret / ret.sum()

def autosharp(image: np.ndarray):
    """sharpens an image based on average/stddev
    brightness and a sigmoid scaling factor"""
    ret = image
    center = ret.max()
    std = ret.std()
    sigmoid = lambda i : 255 * (1 / (1 + np.exp(-(std*i) + center - 5*std)))

    return sigmoid(ret)

# get params
curr = os.getcwd()
path = input("Input image path (current path: '" + curr + "'): ")
val = input("Input blurring kernel size (default: 5): ")
size = 5
if val != "":
    size = int(val)

# the `0` argument reads the file as grayscale
image = cv2.imread(path, 0)

# resize image that's too large
desired_width = 400
width = int(desired_width)
height = int(image.shape[0] * (desired_width / image.shape[1]))

resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

"""
# gaussian blurring

blur = filter(resized, [
    [1, 4,  7,  4,  1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4,  7,  4,  1]
], 1/273)
"""
blur = filter(resized, gaussian(3, size))

edge = filter(blur, [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

display('Black-and-White', resized)
display('Blurred', blur)
display('Edge-detected', edge)
display('Sharpened', autosharp(edge))
