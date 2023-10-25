# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:31:43 2023

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
import time
from scipy import ndimage
import cv2
from scipy.spatial.distance import cdist

def calculate_orientation(num_labels, labeled_image):
    orientations = []

    for label in range(1, num_labels):
        # Extract the boundary points of the labeled region
        points = np.column_stack(np.where(labeled_image == label))

        if len(points) < 2:
            # Skip small regions with less than 2 points
            orientations.append(None)
        else:
            # Compute the covariance matrix
            cov_matrix = np.cov(points, rowvar=False)

            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort the eigenvectors by eigenvalues
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]

            # The major axis is given by the first eigenvector
            major_axis = eigenvectors[:, 0]

            # Calculate the angle of the major axis (in degrees)
            angle_degrees = np.arctan2(major_axis[1], major_axis[0]) * 180 / np.pi

            orientations.append(angle_degrees)

    return orientations

def separate_bacteria(image):
    min_area_threshold = 5
    
    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    orientations = calculate_orientation(num_labels, labeled_image)
    
    # Create an array to keep track of valid labels
    valid_labels = np.zeros(num_labels, dtype=bool)
    
    for label in range(1, num_labels):
        left, top, width, height, area = stats[label]
    
        if area >= min_area_threshold:
            valid_labels[label] = True
    
    # Create a new labeled image with only the valid components
    filtered_labeled_image = np.zeros_like(labeled_image)
    
    for label in range(1, num_labels):
        if valid_labels[label]:
            filtered_labeled_image[labeled_image == label] = label
    
    # Filter out centroids and orientations for valid labels
    filtered_centroids = [centroids[label] for label in range(1, num_labels) if valid_labels[label]]
    filtered_orientations = [orientations[label - 1] for label in range(1, num_labels) if valid_labels[label]]
            
    return filtered_labeled_image, filtered_centroids, filtered_orientations

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\brightfield\20230905-1643'
bf_base_path = base_path + os.sep + 'aligned_images'
ph_base_path = base_path + os.sep + '20230905-1643 phase averages tif'
postitions = os.listdir(bf_base_path)
distances = []
orientations = []
for postition in postitions[:7]:
    bf_positions_path = bf_base_path + os.sep + postition
    ph_positions_path = ph_base_path + os.sep + postition
    timesteps = [int(''.join(filter(str.isdigit, s))) for s in os.listdir(ph_positions_path) if s.endswith('Simple Segmentation.tif')]
    
    for timestep in timesteps:
        bf_fname = bf_positions_path + os.sep + f'{str(timestep).zfill(5)}_BF_Simple Segmentation.tif'
        ph_fname = ph_positions_path + os.sep + f'{str(timestep).zfill(5)}_PH_Simple Segmentation.tif'
        
        bf = tifffile.imread(bf_fname).astype(np.uint8)
        bf[bf!=1] = 0
        bf, centroids_bf, orientations_bf = separate_bacteria(bf)
        ph = tifffile.imread(ph_fname).astype(np.uint8)
        ph[ph!=1] = 0
        ph, centroids_ph, orientations_ph = separate_bacteria(ph)
        
        distance_matrix = cdist(centroids_bf, centroids_ph)
        min_dist = np.argmin(cdist(centroids_bf, centroids_ph), axis=1)
        pairings = [[i, min_dist[i], distance_matrix[i, min_dist[i]]] for i in range(len(min_dist))]
        pairings = [pair for pair in pairings if pair[2]<5]
        for pair in pairings:
            distances.append(centroids_bf[pair[0]]-centroids_ph[pair[1]])
            orientations.append((((orientations_bf[pair[0]]-orientations_ph[pair[1]])+90)%180)-90)
distances = np.array(distances).T

#%%
from matplotlib.colors import LogNorm
fig, ax = plt.subplots()
h = ax.hist2d(distances[0], distances[1], bins=9) #, norm=LogNorm()
fig.colorbar(h[3], ax=ax)
#%%
plt.figure('ph')
plt.imshow(ph)
plt.imshow(bf, alpha=0.5)

#%%
