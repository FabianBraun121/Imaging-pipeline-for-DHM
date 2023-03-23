# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:00:06 2023

@author: SWW-Bc20
"""

import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import clr
import sys
import binkoala
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import warnings
warnings.filterwarnings("ignore")
from utils import connect_to_remote_koala

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\optimizer_bounds'
if not os.path.exists(save_path):
    os.makedirs(save_path)
bound_dists = [5, 2.5, 1.25, 0.6, 0.3, 0.15, 0.075, None]

#%%
########################## start Koala and define functions ##########################
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)

fname = save_path + r"\00000_holo.tif"
host.LoadHolo(fname,1)

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

def evaluate_reconstruction_distance_minus_squared_std(x):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(X, X_pseudoinverse, image_values)
    image_values *= image_values
    return -np.std(image_values)

def evaluate_sharpness_squared_std(x):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(X, X_pseudoinverse, image_values)
    return -sharpness_squared_std(image_values)


def sharpness_squared_std(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Calculate sharpness score as the std gradient magnitude
    sharpness = np.std(grad_mag**4)
    return sharpness

#%%
########################## calculate whole function space ##########################
start = time.time()
x = np.zeros(400)
y = np.zeros(400)
for i in range(400):
    xi = -3+i*0.01
    x[i] = xi
    y[i] = evaluate_sharpness_squared_std([xi])
np.save(save_path+'/x', x)
np.save(save_path+'/y', y)
end = time.time()
print(end-start)
#%%
########################## run test on different optimizing methods, save results ##########################
starts = np.arange(-3,1,0.1)
best = -2.31
results = np.ndarray((len(bound_dists), len(starts)))
duration = np.ndarray((len(bound_dists), len(starts)))
xnfev = np.ndarray((len(bound_dists), len(starts)))
for i, bound_dist in enumerate(bound_dists):
    for j in range(len(starts)):
        if bound_dist==None:
            bounds=None
        else:
            bounds = Bounds(lb=best-bound_dist, ub=best+bound_dist, keep_feasible=False)
        start = time.time()
        res = minimize(evaluate_sharpness_squared_std, [starts[j]], method='Powell', bounds=bounds)
        results[i,j] = res.x[0]
        xnfev[i,j] = res.nfev
        end = time.time()
        duration[i,j] = end-start
        print(j, 'done')
    print(bound_dist, ' done')
np.save(save_path+'/results', results)
np.save(save_path+'/duration', duration)
np.save(save_path+'/xnfev', xnfev)
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
y = np.load(save_path+'/y.npy')
results = np.load(save_path+'/results.npy')
duration = np.load(save_path+'/duration.npy')

#%%
########################## plot ##########################
score = (np.abs(results-x[np.argmin(y)])<0.01).sum(axis=1)
plt.figure()
for i in range(results.shape[0]):
    plt.plot(np.arange(-3,1,0.1), results[i], label=f"tolerance {bound_dists[i]}, score: {score[i]}, avg time: {duration.mean(axis=1)[i]:.2f}, avg evaluations: {xnfev.mean(axis=1)[i]:.2f}")
plt.legend()
plt.title("tolerance with 40 different starting positions")
plt.savefig(save_path+"/optimizing_bounds", dpi=300)
plt.show()
#%%
plt.plot(x,y)