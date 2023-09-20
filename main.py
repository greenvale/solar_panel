
import numpy as np
import sim
import classifier
import os
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image
import cv2
import sim
import math
import matplotlib.pyplot as plt

img_dir_path = './SolarCell-defect-detect/data_BS/test'

labels = ['Break','Cell-faults','Dark-spot','Finger-interuptions','Functional','Material-defect','Microcrack']

sample_size = 10

cam_width = 224

## Create list of images for each label

label_dirs = []

for label in labels:
    label_dirs.append(os.listdir(os.path.join(img_dir_path, label)))

sample_img_paths = []

for label in range(len(labels)):
    curr_samp = 0
    curr_ind = 0
    img_path_row = []
    while curr_samp < sample_size:
        img_path = os.path.join(img_dir_path, labels[label], label_dirs[label][curr_ind])
        try:
            img = Image.open(img_path)
            img_path_row.append(img_path)
            curr_samp += 1
        except:
            print('failed to open image')
        curr_ind += 1
    sample_img_paths.append(img_path_row)

## Panel configuration

D = 0.6

# x, y offsets
dx, dy = 0, 0

vertices = [
    -D+dx, 0, -D+dy,   1.0, 0.0, 0.0,   0.0, 0.0,
    D+dx, 0, -D+dy,   0.0, 1.0, 0.0,   1.0, 0.0,
    D+dx, 0, D+dy,   0.0, 0.0, 1.0,   1.0, 1.0,
    -D+dx, 0, D+dy,   1.0, 1.0, 1.0,   0.0, 1.0,
    ]

indices = [0, 1, 2, 0, 2, 3]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

img_path = sample_img_paths[0][0]
