# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:32:05 2023

@author: SWW-Bc20
"""

"""
Since increasing the order of fit increases the numer of base vectors significantly, the time
increases also a lot. The time for each fit devided by the number of base vectors is approximatly
constant.

"""
import clr
import sys
import binkoala
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\duration_of_plane_subtraction_with_different_fit_orders'

#%%
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')

# Import KoalaRemoteClient
clr.AddReference("LynceeTec.KoalaRemote.Client")
from LynceeTec.KoalaRemote.Client import KoalaRemoteClient

ConfigNumber = 221
# Define KoalaRemoteClient host
host=KoalaRemoteClient()

#Ask IP address
IP = 'localhost'

# Log on Koala - default to admin/admin combo
username = 'admin'
[ret,username] = host.Connect(IP,username,True);
host.Login('admin')
# Open config
host.OpenConfig(ConfigNumber);
host.OpenPhaseWin()
host.OpenIntensityWin()
host.OpenHoloWin()
fname = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\Sample2x2x36_forFabian\2023-02-28 10-06-34\00001\00001_00001\Holograms\00000_holo.tif'
host.LoadHolo(fname,1)
host.SetRecDistCM(-2.3)
host.OnDistanceChange()
host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
ph, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')

def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

#%%
duration = np.zeros(8)
repeats = 10
for i in range(8):
    start = time.time()
    for j in range(repeats):
        a = subtract_plane(ph, i+1)
    end = time.time()
    duration[i] = end-start
    print('degree ', i+1, ' done')
duration /= repeats
np.save(save_path+'/duration', duration)
#%%
duration = np.load(save_path+'/duration.npy')
number_of_base_vectors = np.array([2,5,9,14,20,27,35,44])
plt.figure(1)
plt.bar(np.arange(1,9),duration)
plt.title('time [s] per plane for each order of fit')
#%%
plt.figure(2)
plt.bar(np.arange(1,9),duration/number_of_base_vectors)
plt.title('time [s] per plane devided by number of \n basse vectors for each order of fit')

























