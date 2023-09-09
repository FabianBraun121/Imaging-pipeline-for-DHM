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

def plot_images_with_colorbar(array_2d_top, array_2d_1, array_2d_2, array_2d_3):
    # Create a figure for holding the plots
    fig = plt.figure(figsize=(12, 8))

    # create x and y values
    x = np.arange(array_2d_top.shape[1])
    y = np.arange(array_2d_top.shape[0])
    x, y = np.meshgrid(x, y)

    # Top subplot (3D)
    ax_top = fig.add_subplot(2, 3, 2, projection='3d')
    ax_top.plot_surface(x, y, array_2d_top, cmap='viridis')
    ax_top.set_title('Original phase image', fontsize=20, pad=-50)  # Adjust pad for title position
    ax_top.set_xlabel('X', fontsize=14)
    ax_top.set_ylabel('Y', fontsize=14)
    ax_top.set_zlabel('Z', fontsize=14)

    # First bottom subplot (2D)
    ax1 = fig.add_subplot(2, 3, 4)
    im1 = ax1.imshow(array_2d_1, cmap='viridis')
    ax1.set_title('3rd-degree', fontsize=18)  # Adjust pad for title position
    plt.colorbar(im1, ax=ax1)
    ax1.set_xticks([])  # Remove x ticks
    ax1.set_yticks([])  # Remove y ticks

    # Second bottom subplot (2D)
    ax2 = fig.add_subplot(2, 3, 5)
    im2 = ax2.imshow(array_2d_2, cmap='viridis')
    ax2.set_title('4th-degree', fontsize=18)  # Adjust pad for title position
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks([])  # Remove x ticks
    ax2.set_yticks([])  # Remove y ticks

    # Third bottom subplot (2D)
    ax3 = fig.add_subplot(2, 3, 6)
    im3 = ax3.imshow(array_2d_3, cmap='viridis')
    ax3.set_title('6th-degree', fontsize=18)  # Adjust pad for title position
    plt.colorbar(im3, ax=ax3)
    ax3.set_xticks([])  # Remove x ticks
    ax3.set_yticks([])  # Remove y ticks

    # Adjust layout to be tight
    fig.tight_layout()

    # Show the figure with the plots
    plt.show()


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
    
def plot_images_without_planes(array_2d_top, array_2d_1, array_2d_2, array_2d_3):
    # Create a figure for holding the plots
    fig = plt.figure(figsize=(12, 8))

    # create x and y values
    x = np.arange(array_2d_top.shape[1])
    y = np.arange(array_2d_top.shape[0])
    x, y = np.meshgrid(x, y)

    # Top subplot
    ax_top = fig.add_subplot(2, 3, 2, projection='3d')
    ax_top.plot_surface(x, y, array_2d_top, cmap='viridis')
    ax_top.set_title('original phase image', pad=-20)
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Y')
    ax_top.set_zlabel('Z')

    # First bottom subplot
    ax1 = fig.add_subplot(2, 3, 4, projection='3d')
    ax1.plot_surface(x, y, array_2d_1, cmap='viridis')
    ax1.set_title('3rd-degree', pad=-20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Second bottom subplot
    ax2 = fig.add_subplot(2, 3, 5, projection='3d')
    ax2.plot_surface(x, y, array_2d_2, cmap='viridis')
    ax2.set_title('4th-degree', pad=-20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Third bottom subplot
    ax3 = fig.add_subplot(2, 3, 6, projection='3d')
    ax3.plot_surface(x, y, array_2d_3, cmap='viridis')
    ax3.set_title('6th-degree', pad=-20)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Adjust layout to be tight
    fig.tight_layout()

    # Show the figure with the plots
    plt.show()

def claculate_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return plane

def subtract_plane(field, plane_degree):
    return field - claculate_plane(field, plane_degree)

base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
holo_path = base_path + os.sep + r'data\C11_20230217\2023-02-17 11-13-34\00001\00001_00001\Holograms\00000_holo.tif'
save_path = base_path + os.sep + r'\Graphes_Images\plane_3D'
focus = -2.3
#%%
########################## start Koala and define functions ##########################
ConfigNumber = 219
host = connect_to_remote_koala(ConfigNumber)
host.LoadHolo(holo_path,1)
host.SetRecDistCM(focus)
host.OnDistanceChange()
host.SetUnwrap2DState(True)
#%%
image = host.GetPhase32fImage()
np.save(save_path + os.sep + 'image', image)
#%%
image = np.load(save_path + os.sep + 'image.npy')
#%%
plot_3d(subtract_plane(image, 6))
#%%
plot_images_with_colorbar(image, subtract_plane(image, 3), subtract_plane(image, 4), subtract_plane(image, 6))

#%%
plot_images_without_planes(image, subtract_plane(image, 3), subtract_plane(image, 4), subtract_plane(image, 6))