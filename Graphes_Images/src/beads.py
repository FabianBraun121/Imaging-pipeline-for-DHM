# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:58:33 2023

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
import cv2
from matplotlib.widgets import Slider

def plot_image_series(images):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    time_idx = 0
    imgplot = ax.imshow(images[time_idx])

    fig.colorbar(imgplot)

    def update_image(val):
        nonlocal time_idx
        time_idx = int(val)
        imgplot.set_array(images[time_idx])
        fig.canvas.draw()

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(ax_slider, 'Time', 0, len(images) - 1, valinit=time_idx, valstep=1)
    slider.on_changed(update_image)

    def on_key(event):
        nonlocal time_idx
        if event.key == 'right':
            time_idx = min(time_idx + 1, len(images) - 1)
        elif event.key == 'left':
            time_idx = max(time_idx - 1, 0)
        imgplot.set_array(images[time_idx])
        slider.set_val(time_idx)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

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

def subtract_plane(field, plane_degree, ignore_region=None):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    y = field.reshape(-1)
    
    # If an ignore region is provided, exclude these points from the fitting
    if ignore_region is not None:
        ignore_mask = (X[:, 0] >= ignore_region[0][0]) & (X[:, 0] < ignore_region[0][1]) & \
                      (X[:, 1] >= ignore_region[1][0]) & (X[:, 1] < ignore_region[1][1])
        X_fit = X[~ignore_mask]
        y_fit = y[~ignore_mask]
    else:
        X_fit = X
        y_fit = y

    X_fit = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X_fit)
    
    reg = LinearRegression().fit(X_fit, y_fit)
    
    # Use the full X for prediction (i.e., including the ignored region)
    X_full = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    plane = reg.predict(X_full).reshape(field.shape[0],field.shape[1])
    
    return field - plane


def get_rectangle_coordinates(image):
    # Show the image and wait for user to select a rectangle
    cv2.imshow("Select a rectangle", image)
    rect = cv2.selectROI("Select a rectangle", image, False)
    cv2.destroyAllWindows()

    # Extract coordinates of the rectangle
    x, y, w, h = rect
    ymin, ymax = y, y + h
    xmin, xmax = x, x + w
    
    return [[ymin, ymax], [xmin, xmax]]

#%%
base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
holo_path = base_path + os.sep + r'data\beads\2023-04-21 10-03-21\00001_00001\Holograms\00000_holo.tif'
save_path = base_path + os.sep + r'Graphes_Images\beads'
focus = -2.3

#%%
ConfigNumber = 240
host = connect_to_remote_koala(ConfigNumber)
host.LoadHolo(holo_path,1)
host.SetUnwrap2DState(True)
#%%
foci = np.arange(-3.3, -1.3, 0.01)
images = np.zeros((len(foci), 200, 200))
for i, focus in enumerate(foci):
    host.SetRecDistCM(focus)
    host.OnDistanceChange()
    image = host.GetPhase32fImage()
    image_small = image[93:293,387:587]
    image_small = subtract_plane(image_small, 3, ignore_region=[[50, 150], [50, 150]])
    images[i] = image_small
#%%
refractive_index = 0.06
V_bead = 4/3*np.pi*1**3
theoretical_size = V_bead*refractive_index
plt.figure('size integal')
plt.plot(foci, np.sum(images[:,50:150,50:150], axis=(1,2))*0.794/(2*np.pi)*0.13**2)
plt.plot(foci, np.ones_like(foci)*theoretical_size, 'r')

#%%
plot_image_series(images)

#%%
import matplotlib.colors as mcolors


# Generate a colormap with 20 colors using 'viridis'
cmap = plt.cm.get_cmap('viridis')

# Create a list of 20 colors from the colormap
num_colors = 10
colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]
for i in range(10):
    y_max = np.argmax(images[i*20,50:150,50:150])%100
    plt.plot(images[i*20,y_max+50,50:150], colors[i], label=f'{np.round(foci[i*20],2)}')
plt.legend()