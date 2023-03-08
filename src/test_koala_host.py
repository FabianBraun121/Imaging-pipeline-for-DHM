# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:42:35 2023

@author: SWW-Bc20
"""


import clr
import sys
import binkoala

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
[ret,username] = host.Connect(IP,username,True)
host.Login('admin')
# Open config
host.OpenConfig(ConfigNumber)

#%%
import pathlib
host.OpenPhaseWin()
host.OpenIntensityWin()
host.OpenHoloWin()
fname = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\Sample2x2x36_forFabian\2023-02-28 10-06-34\00001\00001_00001\Holograms\00000_holo.tif'
fpath = pathlib.Path(pathlib.PureWindowsPath(fname))
host.LoadHolo(fname,1)
#%%
import numpy as np
host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
test = -np.std(image_values).mean()

#%%
host.SetRecDistCM(-10)
host.OnDistanceChange()
host.SetUnwrap2DState(True)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def _evaluate_reconstruction_distance(x):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\test.bin')
    image_values = subtract_plane(image_values, 1)
    return -np.std(image_values)

from scipy.optimize import minimize
res = minimize(_evaluate_reconstruction_distance, [-1], method='Powell')

#%%
x = np.zeros(40)
y = np.zeros(40)
for i in range(40):
    xi = -2.4+i*0.01
    x[i] = xi
    y[i] = _evaluate_reconstruction_distance([xi])
#%%
plt.plot(x,y)






#%%
def get_width(koala_host):
    return koala_host.GetPhaseWidth()

a = get_width(host)
    

#%%
import ctypes
buffer = (ctypes.c_int * 1000000)()
#%%
host.GetPhaseImage(buffer);

#%%
host.Logout()

#%%
import numpy as np