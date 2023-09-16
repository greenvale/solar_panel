
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

vertices = [
    -0.71, 0, -0.71,   1.0, 0.0, 0.0,   0.0, 0.0,
    0.71, 0, -0.71,   0.0, 1.0, 0.0,   1.0, 0.0,
    0.71, 0, 0.71,   0.0, 0.0, 1.0,   1.0, 1.0,
    -0.71, 0, 0.71,   1.0, 1.0, 1.0,   0.0, 1.0,
    ]

indices = [0, 1, 2, 0, 2, 3]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

img_path = sample_img_paths[0][0]

iteration = 0

# Environment

def callback_fcn(pitch, azi, window_size):
	global iteration
	new_img = None
	new_pitch = pitch
	new_azi = azi + 0.1
	pixels = glReadPixels(0, 0, window_size, window_size, GL_RGBA, GL_FLOAT)
	pixels = pixels[:,:,0:3]
	pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_LINEAR)
	pixels = np.reshape(pixels, [1, 224, 224, 3])

	if iteration % 100 == 0:
		view = Image.fromarray((pixels[0]*255).astype(np.uint8))
		view.save('./SavedImages/Screenshot'+str(iteration)+'.png')

	iteration += 1

	return new_pitch, new_azi, new_img

env = sim.Environment(vertices, indices)
env.set_image(img_path)

env(callback_fcn)