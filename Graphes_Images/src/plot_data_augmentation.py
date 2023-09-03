# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:14:29 2023

@author: SWW-Bc20
"""
import os
import warnings
import importlib

import cv2
import numpy as np
import skimage.transform as trans
from scipy import interpolate
import elasticdeform
import tifffile
import matplotlib.pyplot as plt

def zoomshift(
    I, zoomlevel: float, shiftX: float, shiftY: float, order: int = 0):
    oldshape = I.shape
    I = trans.rescale(I, zoomlevel, mode="edge", multichannel=False, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I, (shiftY, shiftX), order=order)
    i0 = (
        round(I.shape[0] / 2 - oldshape[0] / 2),
        round(I.shape[1] / 2 - oldshape[1] / 2),
    )
    I = I[i0[0] : (i0[0] + oldshape[0]), i0[1] : (i0[1] + oldshape[1])]
    return I

def shift(image, vector, order = 0):
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode="edge", order=order)

    return shifted

def illumination_voodoo(image, num_control_points: int = 5):

    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(
        image,
        np.reshape(
            np.tile(np.reshape(curve, curve.shape + (1,)), (1, image.shape[1])),
            image.shape,
        ),
    )
    # Rescale values to original range:
    newimage = np.interp(
        newimage, (newimage.min(), newimage.max()), (image.min(), image.max())
    )

    return newimage

def gaussian_noise(image):
    sigma = 0.06
    image = image + np.random.normal(
        0, sigma, image.shape
    )  # Add Gaussian noise
    image = (image - np.min(image)) / np.ptp(
        image
    )  # Rescale to 0-1
    return image

def gaussian_blur(image):
    sigma = 2
    image = cv2.GaussianBlur(image, (5, 5), sigma)
    return image

def elastic_deformation(image):
    image = elasticdeform.deform_random_grid(
        image,
        sigma=25,
        points=3,
        mode="nearest",
        prefilter=False,
    )
    return image

def horizontal_flip(image):
    image = np.fliplr(image)
    return image

def vertical_flip(image):
    image = np.flipud(image)
    return image

def rotation_90(image):
    image = trans.rotate(image, 90, mode="edge")
    return image

def rotation(image):
    image = trans.rotate(image, 5, mode="edge")
    return image

def zoom(image):
    image =  zoomshift(image, 1.15, 0, 0)
    return image

def plot_images_and_difference(images):
    num_images = 4
    # Create a figure and subplots
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    
    lw = 2
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].spines['top'].set_linewidth(lw)
        axes[i].spines['bottom'].set_linewidth(lw)
        axes[i].spines['left'].set_linewidth(lw)
        axes[i].spines['right'].set_linewidth(lw)
        
    # Remove ticks and titles from all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
#%%

b_image = tifffile.imread(r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001\pos00001cha1fra00001.tif')[200:400,200:400]
b_image = (b_image - np.min(b_image)) / np.ptp(b_image)
images = [b_image]
images.append(horizontal_flip(images[-1]))
images.append(vertical_flip(images[-1]))
images.append(rotation_90(images[-1]))
images.append(zoom(images[-1]))
images.append(gaussian_blur(images[-1]))
images.append(gaussian_noise(images[-1]))
images.append(elastic_deformation(images[-1]))
#%%
plot_images_and_difference(images[4:])
plot_images_and_difference(images[:4])
