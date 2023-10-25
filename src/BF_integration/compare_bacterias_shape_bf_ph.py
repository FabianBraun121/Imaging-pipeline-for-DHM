# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:27:23 2023

@author: SWW-Bc20
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
from scipy.ndimage import center_of_mass
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation
import time
from scipy import ndimage
import cv2
from scipy.spatial.distance import cdist

class ImageViewer:
    def __init__(self, image_stack):
        self.image_stack = image_stack
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.img = self.ax.imshow(self.image_stack[self.current_index])
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

def match_bacteria(bf, ph):
    bf_centroid_pairs = {label: center_of_mass(bf, bf, label) for label in np.unique(bf) if label != 0}
    ph_centroid_pairs = {label: center_of_mass(ph, ph, label) for label in np.unique(ph) if label != 0}
    
    # Create a list of labels in both images
    bf_labels = list(bf_centroid_pairs.keys())
    ph_labels = list(ph_centroid_pairs.keys())
    
    if len(bf_labels)==0 or len(ph_labels)==0:
        return np.zeros_like(bf), np.zeros_like(ph)
    
    # Calculate distances between centroids
    distances = cdist([bf_centroid_pairs[label] for label in bf_labels],
                      [ph_centroid_pairs[label] for label in ph_labels])
    
    # Initialize a dictionary to store region pairs
    region_pairs = {}
    
    # Pair the regions based on minimum distance
    for i, bf_label in enumerate(bf_labels):
        min_distance = np.min(distances[i])
        if min_distance <= 5:
            closest_ph_label = ph_labels[np.argmin(distances[i])]
            region_pairs[bf_label] = closest_ph_label
    
    bf_remapped = np.zeros_like(bf)
    ph_remapped = np.zeros_like(ph)
    
    # Remap the labeling starting from 1 with the same region label
    for i, (bf_label, ph_label) in enumerate(region_pairs.items(), 1):
        bf_remapped[bf == bf_label] = i
        ph_remapped[ph == ph_label] = i
    
    return bf_remapped, ph_remapped

def dilute_bacteria(mask):
    diluted_mask = dilation(mask)
    boundaries = diluted_mask-mask
    boundaries[mask!=0] = 0
    return mask + boundaries
        

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\E_coli_steady_state_100_pos'
bf_path = base_path + os.sep + 'Bf_images'
ph_path = base_path + os.sep + 'Ph_images'
bf_mask_fnames = [s for s in os.listdir(bf_path) if s.endswith('Segmentation.tif')]
ph_mask_fnames = [s for s in os.listdir(ph_path) if s.endswith('Segmentation.tif')]
bf_image_fnames = [s for s in os.listdir(bf_path) if s.endswith('BF.tif')]
ph_image_fnames = [s for s in os.listdir(ph_path) if s.endswith('Ph.tif')]

num_dilations = 9
ph_bacteria_sizes = []
bf_bacteria_sizes = np.zeros((0, num_dilations+1))
mask_image_stack = []
bf_image_stack = []
ph_image_stack = []
for n in range(len(bf_mask_fnames)):
    bf_mask = tifffile.imread(bf_path + os.sep + bf_mask_fnames[n]).astype(np.uint8)
    ph_mask = tifffile.imread(ph_path + os.sep + ph_mask_fnames[n]).astype(np.uint8)
    bf_image_stack.append(tifffile.imread(bf_path + os.sep + bf_image_fnames[n]))
    ph_image_stack.append(tifffile.imread(ph_path + os.sep + ph_image_fnames[n]))
    
    bf_cores, _ = ndimage.label(bf_mask==1)
    ph_cores, _ = ndimage.label(ph_mask==1)
    
    bf_remapped, ph_remapped = match_bacteria(bf_cores, ph_cores)
    
    distance = distance_transform_edt(ph_remapped)
    ph_bacteria = watershed(-distance, ph_remapped, mask=(ph_mask<=2))
    ph_bacteria_sizes += list(np.bincount(ph_bacteria.ravel())[1:])
    
    sizes = [np.bincount(bf_remapped.ravel())[1:]]
    for i in range(num_dilations):
        bf_remapped = dilute_bacteria(bf_remapped)
        sizes.append(np.bincount(bf_remapped.ravel())[1:])
        if i == 4:
            mask_image_stack.append(np.where(bf_remapped!=0,1,0)-np.where(ph_bacteria!=0,1,0))
    bf_bacteria_sizes = np.vstack((bf_bacteria_sizes, np.array(sizes).T))
    
    
#%%
viewer = ImageViewer(mask_image_stack)
plt.show()
#%%
viewer = ImageViewer(bf_image_stack)
plt.show()
viewer = ImageViewer(ph_image_stack)
plt.show()

#%%
from skimage.measure import regionprops
regions = regionprops(ph_bacteria)

