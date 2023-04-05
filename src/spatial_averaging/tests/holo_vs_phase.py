# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:59:27 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from utils import connect_to_remote_koala, Open_Directory, get_result_unwrap
import binkoala
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage


start = time.time()
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)
#%%
default_dir = r'Q:\SomethingFun' 
base_dir = Open_Directory(default_dir, "Open a scanning directory")
save_base_folder = base_dir + " phase averages"
if not os.path.exists(save_base_folder):
    os.makedirs(save_base_folder)
all_loc = [ f.name for f in os.scandir(base_dir) if f.is_dir()]
timestamps = len(os.listdir(base_dir+os.sep+all_loc[0]+os.sep+"00001_00001\Holograms"))

#%%
from sklearn.preprocessing import PolynomialFeatures

def _X_plane(plane_fit_order):
    ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
    ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
    X1, X2 = np.mgrid[:800, :800]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    return PolynomialFeatures(degree=plane_fit_order, include_bias=True).fit_transform(X)

def _X_plane_pseudoinverse(plane_fit_order):
    return np.dot( np.linalg.inv(np.dot(_X_plane(plane_fit_order).transpose(), _X_plane(plane_fit_order))), _X_plane(plane_fit_order).transpose())

def _subtract_plane(field, plane_fit_order):
    theta = np.dot(_X_plane_pseudoinverse(plane_fit_order), field.reshape(-1))
    plane = np.dot(_X_plane(plane_fit_order), theta).reshape(field.shape[0], field.shape[1])
    return field-plane, plane

#%%
l = np.random.randint(len(all_loc))
loc = all_loc[l]
loc_dir = base_dir+os.sep+loc
positions = [ f.name for f in os.scandir(loc_dir) if f.is_dir() ]
pos = positions[np.random.randint(len(positions))]
timestep = np.random.randint(timestamps)
fname_holo= loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(timestep).zfill(5) + "_holo.tif"
host.LoadHolo(fname_holo,1)
host.SetUnwrap2DState(True)
host.SetRecDistCM(-2.3)
host.OnDistanceChange()
cache = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file_ph.bin'
host.SaveImageFloatToFile(4,cache,True)
ph_holo_, __header1 = binkoala.read_mat_bin(cache)
ph_holo, plane_holo1 = _subtract_plane(ph_holo_, 2)
ph_holo, plane_holo2 = _subtract_plane(ph_holo_, 5)

fname_phase_image = loc_dir + os.sep + pos + os.sep + f'Phase\\Float\\Bin\\{str(timestep).zfill(5)}_phase.bin'
ph_phase_image_, __header2 = binkoala.read_mat_bin(fname_phase_image)
ph_phase_image, plane_phase_image = _subtract_plane(ph_phase_image_, 5)

#%%
from scipy import ndimage
ph_holo__ = ndimage.shift(np.real(ph_holo), shift=(-3,0), mode='wrap')
#%%
width1 = (int)(__header1["width"])
height1 = (int)(__header1["height"])
px_size1 = (float)(__header1["px_size"])
hconv_read_holo = (float)(__header1["hconv"])
unit_code1 = (int)(__header1["unit_code"])
width2 = (int)(__header2["width"])
height2 = (int)(__header2["height"])
px_size2 = (float)(__header2["px_size"])
hconv_read_phase_image = (float)(__header2["hconv"])
unit_code2 = (int)(__header2["unit_code"])
#%%
binkoala.write_mat_bin('C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file_ph_holo.bin', ph_holo, width1, height1, px_size1, hconv_read_holo, unit_code1)
binkoala.write_mat_bin('C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file_phase_image.bin', ph_holo, width1, height1, px_size1, hconv_read_phase_image, unit_code1)