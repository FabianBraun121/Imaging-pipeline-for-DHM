# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:52:28 2023

@author: fabia
"""
import os
os.chdir(os.path.dirname(__file__))
import binkoala
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import skimage.transform as trans
from skimage.registration import phase_cross_correlation
import time
from scipy import ndimage


def zoom(I, zoomlevel):
    oldshape = I.shape
    I_zoomed = np.zeros_like(I)
    I = trans.rescale(I, zoomlevel, mode="edge")
    if zoomlevel<1:
        i0 = (
            round(oldshape[0]/2 - I.shape[0]/2),
            round(oldshape[1]/2 - I.shape[1]/2),
        )
        I_zoomed[i0[0]:i0[0]+I.shape[0], i0[1]:i0[1]+I.shape[1]] = I
        return I_zoomed
    else:
        I = trans.rescale(I, zoomlevel, mode="edge")
        i0 = (
            round(I.shape[0] / 2 - oldshape[0] / 2),
            round(I.shape[1] / 2 - oldshape[1] / 2),
        )
        I = I[i0[0] : (i0[0] + oldshape[0]), i0[1] : (i0[1] + oldshape[1])]
        return I

def evaluate_error(image1, image2, rotation, zoomlevel):
    im = trans.rotate(image2, rotation, mode="edge")
    im = zoom(im, zoomlevel)
    shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10, normalization=None)
    return error

def grid_search(image1, image2, x_mid, y_mid, x_length, y_length):
    # Initialize the initial grid boundaries.
    x_start, x_end = x_mid - x_length/2, x_mid + x_length/2
    y_start, y_end = y_mid - y_length/2, y_mid + y_length/2
    
    count = 0
    while count in range(4):
        # Create a grid based on the current boundaries.
        x_values = np.linspace(x_start, x_end, 5)
        y_values = np.linspace(y_start, y_end, 5)
        
        # Initialize variables to track the minimum and its location.
        min_value = float('inf')
        min_x, min_y = None, None
        
        # Evaluate the function at each point in the grid.
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                if (i+j)%2==0:
                    value = evaluate_error(image1, image2, x, y)
                    if value < min_value:
                        min_value = value
                        min_x, min_y = x, y
        
        # Check if the minimum is at the edge or in the middle.
        if (
            min_x == x_start or min_x == x_end or
            min_y == y_start or min_y == y_end
        ):
            # If the minimum is at the edge, expand the search space.
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
        else:
            count += 1
            # If the minimum is in the middle, reduce the grid size.
            x_length /= 3
            y_length /= 3
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
    return min_x,min_y

def gradient_squared(image):
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    return (grad_x**2+grad_y**2)

base_path = r'C:\Users\fabia\Documents\GitHub\Imaging-pipeline-for-DHM\data\E_coli_steady_state_100_pos'
save_folder = r'C:\Users\fabia\Documents\GitHub\Imaging-pipeline-for-DHM\data\E_coli_steady_state_100_pos\Aligned_images\E_coli_1430'
image_nums = np.arange(1,101)
rots = []
zooms = []
shift_vectors = []
for image_num in image_nums:
    start = time.time()
    bf_path = base_path + os.sep + 'E_coli_1430' + os.sep + f'{str(image_num).zfill(5)}' + os.sep + '00000_BF.tif'
    ph_path = base_path + os.sep + 'E_coli_1430 phase averages' + os.sep +  f'{str(image_num).zfill(5)}' + os.sep + 'ph_timestep_00000.bin'
    
    bf = tifffile.imread(bf_path)[512:1536,512:1536]
    bf = np.fliplr(bf)
    bf = trans.rotate(bf, -90, mode="edge")
    ph, _ = binkoala.read_mat_bin(ph_path)
    ph_ = np.zeros(bf.shape)
    ph_[:ph.shape[0], :ph.shape[1]] = ph
    
    bf_ = gradient_squared(bf)
    ph_ = gradient_squared(ph_)
    rot, zoomlevel = grid_search(ph_, bf_, 0, 0.9, 0.5, 0.1)
    bf_rz = zoom(trans.rotate(bf_, rot, mode="edge"),zoomlevel)
    shift_measured = phase_cross_correlation(ph_, bf_rz, upsample_factor=10, normalization=None)[0]
    shift_vector = (shift_measured[0], shift_measured[1])
    
    bf_out = ndimage.shift(zoom(trans.rotate(bf, rot, mode="edge"),zoomlevel), shift_vector)[:ph.shape[0], :ph.shape[1]]
    fname = save_folder + os.sep + f'{str(image_num).zfill(5)}_BF.tif'
    tifffile.imwrite(fname, bf_out)
    
    rots.append(rot)
    zooms.append(zoomlevel)
    shift_vectors.append(shift_vector)
    
    print(f'image {image_num} done in {time.time()-start} seconds')

#%%
plt.figure('bf')
plt.imshow(zoom(trans.rotate(bf_, 0, mode="edge"),0.92))
plt.figure('ph')
plt.imshow(ph_)
#%%
errors = np.zeros((11,11))
zoomlevels = np.linspace(0.89,0.91,11)
rotations = np.linspace(-0.5,0.5,11)

for i, rotation in enumerate(rotations):
    for j, zoomlevel in enumerate(zoomlevels):
        bf__ = trans.rotate(bf_, rotation, mode="edge")
        bf__ = zoom(bf__, zoomlevel)
        shift_measured, error, phasediff = phase_cross_correlation(ph_, bf__, upsample_factor=10, normalization=None)
        print(shift_measured)
        errors[i,j] = error
plt.imshow(errors)
min_index = np.unravel_index(np.argmin(errors), errors.shape)
print(rotations[min_index[0]])
print(zoomlevels[min_index[1]])
#%%
min_index = np.unravel_index(np.argmin(errors), errors.shape)
bf__ = zoom(trans.rotate(bf_, rot, mode="edge"),zoomlevel)
shift_measured = phase_cross_correlation(ph_, bf__, upsample_factor=10, normalization=None)[0]
shift_vector = (shift_measured[0], shift_measured[1])
bf__ = ndimage.shift(bf__, shift_vector)        

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class ImageViewer:
    def __init__(self, image_stack):
        self.image_stack = image_stack
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.img = self.ax.imshow(self.image_stack[self.current_index], cmap='gray')
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_image)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_image)

        self.connect_key_events()

    def connect_key_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.next_image(None)
        elif event.key == 'left':
            self.prev_image(None)

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.image_stack)
        self.update_image()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.image_stack)
        self.update_image()

    def update_image(self):
        self.img.set_data(self.image_stack[self.current_index])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')
        self.fig.canvas.draw()

image_stack = [ph/ph.max(), bf_out/bf_out.max()]
viewer = ImageViewer(image_stack)
plt.show()


















