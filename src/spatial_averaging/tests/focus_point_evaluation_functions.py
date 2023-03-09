# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:51:04 2023

@author: SWW-Bc20
"""
import os
import clr
import sys
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\focus_point_evaluation_functions'
if not os.path.exists(save_path):
    os.makedirs(save_path)
import binkoala
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
    
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
#%%
def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def evaluate_reconstruction_distance_minus_std(x, degree):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(image_values, degree)
    return -np.std(image_values)

def evaluate_reconstruction_distance_minus_squared_std(x, degree):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(image_values, degree)
    image_values *= image_values
    return -np.std(image_values)
#%%
x = np.zeros(400)
y_std = np.zeros((400, 4))
y_squared_std = np.zeros((400, 4))
for i in range(400):
    for j in range(4):
        xi = -3+i*0.01
        x[i] = xi
        y_std[i,j] = -evaluate_reconstruction_distance_minus_std([xi], j+1)
        y_squared_std[i,j] = -evaluate_reconstruction_distance_minus_squared_std([xi], j+1)
    print(x[i], " done")
np.save(save_path+'/x', x)
np.save(save_path+'/y_std', y_squared_std)
np.save(save_path+'/y_squared_std', y_squared_std)
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
y_std = np.load(save_path+'/y_std.npy')
y_squared_std = np.load(save_path+'/y_squared_std.npy')
#%%
plt.figure("std")
for i in range(4):
    plt.plot(x, y_std[:,i], label=f'plane degree {i+1}')
plt.xlabel("lengths [cm]")
plt.title("std of different reconstruction lengths")
plt.legend()
plt.savefig(save_path+"/std", dpi=300)
plt.show()
#%%
plt.figure("squared std")
for i in range(4):
    plt.plot(x, y_squared_std[:,i], label=f'plane degree {i+1}')
plt.xlabel("lengths [cm]")
plt.title("squared std of different reconstruction lengths")
plt.legend()
plt.savefig(save_path+"/squared_std", dpi=300)
plt.show()





















