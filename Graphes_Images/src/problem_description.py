# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:22:15 2023

@author: SWW-Bc20
"""

import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from utils import connect_to_remote_koala
import time
import scipy.ndimage
from spatial_phase_averaging import SpatialPhaseAveraging


base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
data_path = base_path + r'\data\E10_20230216\2023-02-16 12-14-58\00001'
save_path = base_path + r'\Graphes_Images\problem_description'
fname_example_holo = data_path + r'\00001_00001\Holograms\00000_holo.tif'


#%%
########################## start Koala and define functions ##########################
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)

def generate_X_and_pseudoinverse(plane_degree):
    X1, X2 = np.mgrid[:800, :800]
    X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=True).fit_transform(X)
    pseudoinverse = np.dot( np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
    return X, pseudoinverse

X, X_pseudoinverse = generate_X_and_pseudoinverse(3)

def subtract_plane(X, pseudoinverse, field):
    theta = np.dot(pseudoinverse, field.reshape(-1))
    plane = np.dot(X, theta).reshape(field.shape[0], field.shape[1])
    return field-plane

def evaluate_std_sobel_squared(x):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(X, X_pseudoinverse, image_values)
    return -std_sobel_squared(image_values)

def std_sobel_squared(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    grad_mag = grad_x ** 2 + grad_y ** 2
    
    # Calculate sharpness score as the std gradient magnitude
    sharpness = np.std(grad_mag)
    return sharpness

#%%
host.LoadHolo(fname_example_holo,1)

#%%
########################## calculate whole function space ##########################
start = time.time()
x = np.zeros(400)
y = np.zeros(400)
for i in range(400):
    xi = -3+i*0.01
    x[i] = xi
    y[i] = evaluate_std_sobel_squared([xi])
np.save(save_path+'/x', x)
np.save(save_path+'/y', y)
end = time.time()
#%%
x = np.load(save_path+'/x.npy')
y = np.load(save_path+'/y.npy')

host.SetRecDistCM(-2.31)
host.OnDistanceChange()
host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
image_values = subtract_plane(X, X_pseudoinverse, image_values)
plt.figure('In Focus')
plt.title('bacteria in focus')
plt.imshow(image_values)
plt.savefig(save_path+"/in_focus", dpi=300)

host.SetRecDistCM(-2.0)
host.OnDistanceChange()
host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
image_values = subtract_plane(X, X_pseudoinverse, image_values)
plt.figure('Out of Focus')
plt.title('bacteria out of focus')
plt.imshow(image_values)
plt.savefig(save_path+"/out_of_focus", dpi=300)

plt.figure("sharpness evaluation")
plt.title("function evaluation sharpness")
plt.plot(x,y, label='sharpness =\n-std(sobel_kernel(image(d)))**2)')
plt.plot([-2.31,-2.31],[-0.036, -0.024], 'g')
plt.plot([-2.0,-2.0],[-0.036, -0.024], 'r')
plt.xlabel('focus point distance d [cm]')
plt.legend(fontsize=12)
plt.savefig(save_path+"/function_evaluation_sharpness", dpi=300)
#%%
spa = SpatialPhaseAveraging(data_path, 0, host)
#%%
plt.figure('Averaged Image')
plt.title('averaged image')
plt.imshow(np.angle(spa.cplx_avg))
plt.savefig(save_path+"/averaged_image", dpi=300)