# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:41:28 2023

@author: SWW-Bc20
"""

import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from utils import connect_to_remote_koala
import binkoala
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.ndimage
import random

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\time_processes'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)
fname = save_path + r"\00000_holo.tif"
host.LoadHolo(fname,1)
#%%
def polynomal_extension(X_poly, degrees_left, columns_old):
    if degrees_left<=1:
        return X_poly
    else: 
        columns_new = X_poly.shape[1]
        for i in range(columns_old, columns_new):
            X_poly = np.hstack((X_poly, (X_poly[:,0]*X_poly[:,i]).reshape(-1,1)))
        X_poly = np.hstack((X_poly, (X_poly[:,1]*X_poly[:,columns_new-1]).reshape(-1,1)))
        return polynomal_extension(X_poly, degrees_left-1, columns_new)

def generate_X(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    X = polynomal_extension(X, plane_degree, 0)
    return np.hstack((np.ones(np.size(X1)).reshape(-1,1), X))

def generate_X_and_pseudoinverse(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    X = polynomal_extension(X, plane_degree, 0)
    X = np.hstack((np.ones(np.size(X1)).reshape(-1,1), X))
    pseudoinverse = np.dot( np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
    return X, pseudoinverse

def subtract_plane_by_hand(field, plane_degree):
    X = generate_X(field, plane_degree)
    theta = np.dot(np.dot( np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), field.reshape(-1))
    plane = np.dot(X, theta).reshape(field.shape[0], field.shape[1])
    return field-plane

def subtract_plane_by_hand_X_precomputed(X, pseudoinverse, field, plane_degree):
    theta = np.dot(pseudoinverse, field.reshape(-1))
    plane = np.dot(X, theta).reshape(field.shape[0], field.shape[1])
    return field-plane

def subtract_plane_sklearn(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def sharpness_squared_std(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Calculate sharpness score as the std gradient magnitude
    sharpness = np.std(grad_mag**2)
    return sharpness

def squared_std(image):
    return np.std(image**2)

#%%
start = time.time()
for i in range(100):
    host.SetRecDistCM(random.random())
    host.OnDistanceChange()
    image_values = host.GetIntensity32fImage()
end = time.time()
print('Saving and reading an image in Koala takes',np.round((end-start)*10,1), 'ms per image')
#%%
start = time.time()
for i in range(100):
    host.SetRecDistCM(random.random())
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
end = time.time()
print('Saving and reading an image in Koala takes',np.round((end-start)*10,1), 'ms per image')
#%%
host.SetRecDistCM(-2.3)
host.OnDistanceChange()
host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
start = time.time()
for i in range(100):
    X = generate_X(image_values, 3)
end = time.time()
print('Generating the X plane with self written function takes', np.round((end-start)*10,1), 'ms per image')
#%%
start = time.time()
for i in range(100):
    X1, X2 = np.mgrid[:800, :800]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
end = time.time()
print('Generating the X plane with sklearn function takes',np.round((end-start)*10,1), 'ms per image')
#%%
start = time.time()
for i in range(100):
    image_sklearn = subtract_plane_sklearn(image_values, 3)
end = time.time()
print('Subtracting plane with sklearn function takes', np.round((end-start)*10,1), 'ms per image')
#%%
start = time.time()
for i in range(100):
    image_hand = subtract_plane_by_hand(image_values, 3)
end = time.time()
print('Subtracting plane with self written function takes', np.round((end-start)*10,1), 'ms per image')
#%%
X, pseudoinverse = generate_X_and_pseudoinverse(image_values, 3)
start = time.time()
for i in range(100):
    image_hand_precomputed = subtract_plane_by_hand_X_precomputed(X, pseudoinverse, image_values, 3)
end = time.time()
print('Subtracting plane with self written function and precomputed pseudoinverse takes', np.round((end-start)*10,1), 'ms per image')
#%%
start = time.time()
for i in range(100):
    __ = sharpness_squared_std(image_hand_precomputed)
end = time.time()
print('Sharpness squared std algorithm takes', np.round((end-start)*10,1), 'ms per image')

#%%
start = time.time()
for i in range(100):
    __ = squared_std(image_hand_precomputed)
end = time.time()
print('Image squared std algorithm takes', np.round((end-start)*10,1), 'ms per image')