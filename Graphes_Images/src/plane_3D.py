# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 07:39:35 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
from  pyKoalaRemote import client
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

def connect_to_remote_koala(ConfigNumber):
    # Define KoalaRemoteClient host
    host = client.pyKoalaRemoteClient()
    #Ask IP address
    IP = 'localhost'
    # Log on Koala - default to admin/admin combo
    host.Connect(IP)
    host.Login('admin')
    # Open config
    host.OpenConfig(ConfigNumber)
    host.OpenPhaseWin()
    host.OpenIntensityWin()
    host.OpenHoloWin()
    return host

def plot_3d(array_2d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create x and y values
    x = np.arange(array_2d.shape[1])
    y = np.arange(array_2d.shape[0])
    x, y = np.meshgrid(x, y)

    # 2D array values are used as z values
    z = array_2d

    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane


base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
holo_path = base_path + r'data\C11_20230217\2023-02-17 11-13-34\00001\00001_00001\Holograms\00000_holo.tif'
save_path = base_path + r'\Graphes_Images\plane_3D'
focus = -2.3
#%%
########################## start Koala and define functions ##########################
ConfigNumber = 219
host = connect_to_remote_koala(ConfigNumber)
host.LoadHolo(holo_path,1)
host.SetRecDistCM(focus)
host.OnDistanceChange()
host.SetUnwrap2DState(True)
image = host.GetPhase32fImage()
np.save(save_path + os.sep + 'image', image)
#%%
image = np.load(save_path + os.sep + 'image.npy')
#%%
plot_3d(image)
#%%
plane = image - subtract_plane(image, 5)
plot_3d(subtract_plane(image, 6))
