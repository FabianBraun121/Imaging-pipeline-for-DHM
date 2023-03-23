# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:28:24 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from spatial_phase_averaging import SpatialPhaseAveraging
from utils import connect_to_remote_koala
import binkoala
import time
import numpy as np
import matplotlib.pyplot as plt

holo =  r"F:\F3_20230302\2023-03-02 11-30-06\00003\00001_00001\Holograms\00001_holo.tif"
cache_path = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file'
#%%
ConfigNumber = 224
host = connect_to_remote_koala(ConfigNumber)
host.LoadHolo(holo,1)
#%%
ph = np.ndarray((900,800,800))
amp = np.ndarray((900,800,800))
for i in range(900):
    xi = -45+0.1*i
    host.SetRecDistCM(xi)
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,cache_path+'_ph.bin',True)
    ph_i, __header = binkoala.read_mat_bin(cache_path+'_ph.bin')
    host.SaveImageFloatToFile(2,cache_path+'_amp.bin',True)
    amp_i, __header = binkoala.read_mat_bin(cache_path+'_amp.bin')
    ph[i] = ph_i
    amp[i] = amp_i
#%%

plt.figure()
plt.imshow(np.std((ph[400:450]*ph[400:450]), axis=0))

#%%
plt.figure()
plt.plot(np.arange(-45,45,0.1),np.mean((ph*ph), axis=(1,2)))

#%%
plt.figure()
plt.plot(np.arange(-45,45,0.1),np.std(ph*ph, axis=(1,2)))