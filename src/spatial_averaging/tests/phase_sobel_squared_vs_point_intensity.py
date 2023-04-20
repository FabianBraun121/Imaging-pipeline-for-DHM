# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:06:12 2023

@author: SWW-Bc20
"""
import os
import clr
import sys
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.ndimage
import cv2
from pyKoalaRemote import client
from scipy import ndimage
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging\tests')
from test_utilities import interactive_image_player

#%%


ConfigNumber = 219
# Define KoalaRemoteClient host
host = client.pyKoalaRemoteClient()
host.Connect('localhost')
host.Login('admin')
# Open config
host.OpenConfig(ConfigNumber);
host.OpenPhaseWin()
host.OpenIntensityWin()
host.OpenHoloWin()

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\phase_sobel_squared_vs_point_intensity'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%%
def generate_X_and_pseudoinverse(plane_degree):
    X1, X2 = np.mgrid[:800, :800]
    X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    X =  PolynomialFeatures(degree=plane_degree).fit_transform(X)
    pseudoinverse = np.dot( np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
    return X, pseudoinverse

def subtract_plane_by_hand_X_precomputed(X, pseudoinverse, field):
    theta = np.dot(pseudoinverse, field.reshape(-1))
    plane = np.dot(X, theta).reshape(field.shape[0], field.shape[1])
    return field-plane

def evaluate_sobel_squared_std(gray_image) -> float:
    # Calculate gradient magnitude using Sobel filter
    grad_x = ndimage.sobel(gray_image, axis=0)
    grad_y = ndimage.sobel(gray_image, axis=1)
    # Calculate std squared sobel sharpness score
    return np.std(grad_x ** 2 + grad_y ** 2)

def normalize(function):
    return (function-np.min(function))/(np.max(function)-np.min(function))

def show_image_with_points(image, points):
    # Create a copy of the image to avoid modifying the original array
    image_copy = np.copy(image)
    # Set the marked pixels to red (maximum value)
    for point in points:
        x, y = point
        image_copy[y, x] = np.max(image)*1.5
    # Plot the image with the marked points
    plt.imshow(image_copy)
    plt.show()
#%%
fname = save_path + r"\00000_holo.tif"
host.LoadHolo(fname,1)
host.SetUnwrap2DState(True)
X, pseudoinverse = generate_X_and_pseudoinverse(5)

x = np.zeros(800)
ph_images = np.zeros((x.shape[0],800,800))
int_images = np.zeros((x.shape[0],800,800))
sobel_squared = np.zeros(x.shape)

for i in range(x.shape[0]):
    xi = -2.5+i*0.001
    x[i] = xi
    host.SetRecDistCM(xi)
    host.OnDistanceChange()
    ph_image = host.GetPhase32fImage()
    ph_image = subtract_plane_by_hand_X_precomputed(X, pseudoinverse, ph_image)
    ph_images[i] = ph_image
    sobel_squared[i] = evaluate_sobel_squared_std(ph_image)
    int_image = host.GetIntensity32fImage()
    int_image = subtract_plane_by_hand_X_precomputed(X, pseudoinverse, int_image)
    int_images[i] = int_image
    print(f'distance:{xi} done!')

np.save(save_path+'/x', x)
np.save(save_path+'/ph_images', ph_images)
np.save(save_path+'/int_images', int_images)
np.save(save_path+'/sobel_squared', sobel_squared)
#%%
x = np.load(save_path+'/x.npy')
ph_images = np.load(save_path+'/ph_images.npy')
int_images = np.load(save_path+'/int_images.npy')
sobel_squared = np.load(save_path+'/sobel_squared.npy')
#%%
ideal_focus_image = ph_images[np.argmax(sobel_squared)]
#%%
cut_off = 0.4
potential_points = np.array([np.where(cut_off<ideal_focus_image)[1], np.where(cut_off<ideal_focus_image)[0]]).T
show_image_with_points(ideal_focus_image, potential_points)
#%%
int_functions_of_potential_points = int_images[:,potential_points[:,1], potential_points[:,0]]
int_zero_of_potential_points = np.argmin(np.abs(int_functions_of_potential_points), axis=0)
filtered_int_zero_of_potential_points = int_zero_of_potential_points[int_zero_of_potential_points<700]

#%%
b_xy = [[272,155], [266,212], [481,244], [713,661], [131,475]]
color = ['b', 'g', 'r', 'c', 'm', 'y']
for i, point in enumerate(b_xy):
    f = int_images[:,point[1],point[0]]
    x_ = np.argmin(np.abs(f))
    _y = normalize(f)
    plt.plot(x, f , color[i+1], label=f'point intensity {i}, zero at {x[x_]}')
    _x = np.array([x[x_],x[x_]])
    __y = np.array([np.min(f), np.max(f)])
    plt.plot(_x, __y, color[i+1])
_y = normalize(sobel_squared)
plt.plot(x, _y,color[0], label=f'sobel_squared, max at {x[np.argmax(_y)]}')
_x = np.array([x[np.argmax(_y)],x[np.argmax(_y)]])
plt.plot(_x, __y, color[0])
plt.legend()
#%%

plt.plot(x, int_images[:,155,272])

#%%
interactive_image_player(int_images)