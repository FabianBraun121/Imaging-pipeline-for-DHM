# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:39:35 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import skimage.transform as trans
from skimage.registration import phase_cross_correlation
from scipy import ndimage
import time

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
    shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10)
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

base_path =r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20231018_ecoli_EZ'
pc_base_path = base_path + os.sep + '20231018_1600_EZ_Culture1'
ph_base_path = base_path + os.sep + '20231018_1600_EZ_Culture1 phase averages'
postitions = os.listdir(pc_base_path)

rots = []
zooms = []
shift_vectors = []
image_stack = []

for postition in postitions[1:2]:
    start = time.time()
    pc_positions_path = pc_base_path + os.sep + postition
    ph_positions_path = ph_base_path + os.sep + postition
    
    pc_fname = pc_positions_path + os.sep + f'{str(0).zfill(5)}_BF.tif'
    ph_fname = ph_positions_path + os.sep + f'ph_timestep_{str(0).zfill(5)}.tif'
    
    pc = tifffile.imread(pc_fname)
    pc = np.fliplr(pc)
    pc = trans.rotate(pc, -90, mode="edge")[512:1536,512:1536]
    ph = tifffile.imread(ph_fname)
    ph_ = np.zeros(pc.shape)
    ph_[:ph.shape[0], :ph.shape[1]] = ph
    
    pc_ = gradient_squared(pc)
    ph_ = gradient_squared(ph_)
    
    rot, zoomlevel = grid_search(ph_, pc_, 0, 0.905, 2, 0.2)
    pc_rz = zoom(trans.rotate(pc_, rot, mode="edge"),zoomlevel)
    shift_measured = phase_cross_correlation(ph_, pc_rz, upsample_factor=10)[0]
    shift_vector = (shift_measured[0], shift_measured[1])
    pc_out = ndimage.shift(zoom(trans.rotate(pc, rot, mode="edge"),zoomlevel), shift_vector)[:ph.shape[0], :ph.shape[1]]
    
    rots.append(rot)
    zooms.append(zoomlevel)
    shift_vectors.append(shift_vector)
    image_stack.append((ph-ph.min())/(ph.max()-ph.min()))
    image_stack.append((pc_out-pc_out.min())/(pc_out.max()-pc_out.min()))
print(start-time.time())    
#%%
zooms = np.linspace(0.5,1.2,30)
errors = []
for i in range(len(zooms)):
    pc_rz = zoom(pc_,zooms[i])
    shift_measured, error, phasediff  = phase_cross_correlation(ph_, pc_rz, upsample_factor=10)
    errors.append(error)
#%%
pc_rz = zoom(pc_, 0.7413793103448276)
shift_measured, error, phasediff  = phase_cross_correlation(ph_, pc_rz, upsample_factor=10)
shift_vector = (shift_measured[0], shift_measured[1])
pc_out = ndimage.shift(zoom(pc,0.7413793103448276), shift_vector)[:ph.shape[0], :ph.shape[1]]

#%%
viewer = ImageViewer(image_stack)
plt.show()
