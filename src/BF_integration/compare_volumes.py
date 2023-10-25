# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:37:58 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import numpy as np
from skimage.measure import regionprops
import tifffile
from scipy.spatial.distance import cdist
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import skeletonize, medial_axis
import matplotlib.pyplot as plt
import math

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

def V_ellipsoid(major_axis, minor_axis):
    return 4/3 * np.pi * (0.5*major_axis) * (0.5*minor_axis)**2

def V_rod(major_axis, minor_axis):
    V_sphere =  4/3 * np.pi * (0.5*minor_axis)**3
    V_cylinder = np.pi * (0.5*minor_axis)**2 * (0.5*(major_axis - minor_axis))
    return V_sphere + V_cylinder

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\E_coli_steady_state_100_pos'
bf_path = base_path + os.sep + 'Bf_images'
ph_path = base_path + os.sep + 'Ph_images'
bf_mask_fnames = [s for s in os.listdir(bf_path) if s.endswith('Segmentation.tif')]
ph_mask_fnames = [s for s in os.listdir(ph_path) if s.endswith('Segmentation.tif')]

v_ellipsoids = []
v_rods = []
for n in range(len(bf_mask_fnames))[:2]:
    bf_mask = tifffile.imread(bf_path + os.sep + bf_mask_fnames[n]).astype(np.uint8)
    ph_mask = tifffile.imread(ph_path + os.sep + ph_mask_fnames[n]).astype(np.uint8)
    
    bf_cores, _ = label(bf_mask==1)
    ph_cores, _ = label(ph_mask==1)
    
    bf_remapped, ph_remapped = match_bacteria(bf_cores, ph_cores)
    
    distance = distance_transform_edt(ph_remapped)
    ph_bacteria = watershed(-distance, ph_remapped, mask=(ph_mask<=2))
    regions = regionprops(ph_bacteria)
    for region in regions:
        v_ellipsoids.append(V_ellipsoid(region.major_axis_length, region.minor_axis_length))
        v_rods.append(V_rod(region.major_axis_length, region.minor_axis_length))

#%%
regions = regionprops(ph_bacteria)

fig, ax = plt.subplots()
ax.imshow(ph_bacteria, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()

#%%
im = np.where(ph_bacteria!=0,1,0)
plt.figure('image')
plt.imshow(im+skeletonize(im))
plt.figure('skeletonize')
plt.imshow(skeletonize(im))
plt.figure('medial_axis')
plt.imshow(medial_axis(im))
#%%
im = np.where(ph_bacteria==1,1,0)
plt.figure('image')
plt.imshow(im)
plt.figure('skeletonize')
plt.imshow(skeletonize(im))
plt.figure('medial_axis')
plt.imshow(medial_axis(im))


#%%
# https://stackoverflow.com/questions/4555682/determine-the-midline-of-a-bent-elongated-region
def find_midline(binary_image, max_iterations=100):
    # Calculate the distance transform of the inside of the object
    distance_map = distance_transform_edt(binary_image)
    
    # Initialize the midline with a straight line across the object
    midline = initialize_midline(binary_image)
    
    for iteration in range(max_iterations):
        # Create an empty list to store the new midline points
        new_midline = []
        
        for point in midline:
            # Calculate the force pushing the midline away from the border
            gradient = calculate_gradient(distance_map, point)
            # Calculate a force to maintain smoothness (simplified as moving towards the previous point)
            smooth_force = calculate_smooth_force(midline, point)
            
            # Combine the forces and update the point on the midline
            new_point = point + gradient + smooth_force
            new_midline.append(new_point)
        
        midline = new_midline

    return midline

def calculate_gradient(distance_map, point):
    # Calculate the force pushing the midline away from the border
    gradient = np.gradient(distance_map, axis=0)
    return gradient[point]

def calculate_smooth_force(midline, point):
    # Calculate a force to maintain smoothness (simplified as moving towards the previous point)
    if len(midline) > 1:
        previous_point = midline[-2]
        smooth_force = (previous_point - point) / 2
    else:
        smooth_force = np.array([0, 0])
    return smooth_force

def initialize_midline(binary_image):
    # Initialize the midline with a straight line across the object
    props = regionprops(binary_image)[0]
    
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.sin(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    height, width = binary_image.shape
    midline = [(int(height / 2), x) for x in range(width) if binary_image[int(height / 2), x] == 1]
    return midline

midline = find_midline(np.where(ph_bacteria==2,1,0), max_iterations=100)

#%%
im = np.where(ph_bacteria==2,1,0)
distance_map = distance_transform_edt(im)
gradient = np.gradient(distance_map, axis=0)

