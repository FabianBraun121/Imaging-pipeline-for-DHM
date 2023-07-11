# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:06:49 2023

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
import matplotlib.gridspec as gridspec


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

def unity_based_normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

#%%
base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
holo_path = base_path + os.sep + r'data\focus_functions\2023-04-21 10-03-21\00001_00001\Holograms\00000_holo.tif'
save_path = base_path + os.sep + r'Graphes_Images\focus_functions'
#%%
focus = -2.3
ConfigNumber = 240
host = connect_to_remote_koala(ConfigNumber)
#%%
host.LoadHolo(holo_path,1)
host.SetUnwrap2DState(True)
foci = np.linspace(-3.3, -1.3, 200)
ph_images = np.zeros((len(foci), 100, 100))
amp_images = np.zeros((len(foci), 100, 100))
for i, focus in enumerate(foci):
    host.SetRecDistCM(focus)
    host.OnDistanceChange()
    ph_image = host.GetPhase32fImage()
    amp_image = host.GetIntensity32fImage()
    
    ph_image_small = ph_image[93:293,387:587]
    ph_image_small = subtract_plane(ph_image_small, 3, ignore_region=[[50, 150], [50, 150]])
    ph_images[i] = ph_image_small[50:150,50:150]
    
    amp_image_small = amp_image[93:293,387:587]
    # amp_image_small = subtract_plane(amp_image_small, 3, ignore_region=[[50, 150], [50, 150]])
    amp_images[i] = amp_image_small[50:150,50:150]
    
#%%
plot_image_series(amp_images)

#%%

# three images (columns)
fig = plt.figure()
gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 0.05, 1,0.2], width_ratios=[1, 1, 1, 0.1], wspace=0.01, hspace=0.01, top=0.95, bottom=0.05, left=0.17, right=0.845)

amp = [amp_images[0], amp_images[amp_images.shape[0]//2], amp_images[-1]]
ph = [ph_images[0], ph_images[ph_images.shape[0]//2], ph_images[-1]]
d = [foci[0], foci[foci.shape[0]//2], foci[-1]]

for i in range(0, 3):
    f_ax1 = fig.add_subplot(gs[0, i])
    f_ax1.set_xticks([])    # Remove x-axis ticks
    f_ax1.set_yticks([])    # Remove y-axis ticks
    f_ax1.set_xticklabels([])  # Remove x-axis labels
    im1 = f_ax1.imshow(amp[i], vmin=np.min(amp), vmax=np.max(amp))
    f_ax1.set_title(f'{round(d[i],1)} cm', fontsize=12)

    f_ax2 = fig.add_subplot(gs[1, i])
    f_ax2.set_xticks([])    # Remove x-axis ticks
    f_ax2.set_yticks([])    # Remove y-axis ticks
    f_ax2.set_xticklabels([])  # Remove x-axis labels
    im2 = f_ax2.imshow(ph[i], vmin=np.min(ph), vmax=np.max(ph))
    if i == 0:
        f_ax1.set_ylabel('amplitude', fontsize=12)
        f_ax2.set_ylabel('phase', fontsize=12)

# Add colorbars on the right side
cbar_ax1 = fig.add_subplot(gs[0, -1])
fig.colorbar(im1, cax=cbar_ax1)

cbar_ax2 = fig.add_subplot(gs[1, -1])
fig.colorbar(im2, cax=cbar_ax2)

# Add the plots to the third row
ax3 = fig.add_subplot(gs[3, :])
std_amp = np.std(amp_images, axis=(1,2))
std_ph = np.std(ph_images, axis=(1,2))
combined = -np.std(ph_images, axis=(1,2))/np.std(amp_images, axis=(1,2))
ax3.plot(foci, unity_based_normalization(std_amp), 'b', label='std_amp')
ax3.plot(np.ones(2)*foci[np.argmin(std_amp)], np.array([0,1]), 'b--')
ax3.plot(foci, unity_based_normalization(std_ph), 'g', label='std_ph')
ax3.plot(foci, unity_based_normalization(combined), color='orange', label='combined')
ax3.plot(np.ones(2)*foci[np.argmin(combined)], np.array([0,1]), '--', color='orange')
ax3.set_xlabel('focus distance [cm]', fontsize=12)
ax3.legend()

plt.subplots_adjust(wspace=0, hspace=0)  # Adjusts the space between the subplots
plt.show()

#%%

# five images (columns)
fig = plt.figure(figsize=(11.86,5.7))
gs = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 0.05, 1,0.2], width_ratios=[1, 1, 1, 1, 1, 0.1], wspace=0.01, hspace=0.01, top=0.95, bottom=0.05, left=0.17, right=0.845)

amp = [amp_images[0], amp_images[amp_images.shape[0]//4], amp_images[amp_images.shape[0]//2], amp_images[amp_images.shape[0]//4*3], amp_images[-1]]
ph = [ph_images[0], ph_images[ph_images.shape[0]//4], ph_images[ph_images.shape[0]//2], ph_images[ph_images.shape[0]//4*3], ph_images[-1]]
d = [foci[0], foci[foci.shape[0]//4], foci[foci.shape[0]//2], foci[foci.shape[0]//4*3], foci[-1]]

for i in range(0, 5):
    f_ax1 = fig.add_subplot(gs[0, i])
    f_ax1.set_xticks([])    # Remove x-axis ticks
    f_ax1.set_yticks([])    # Remove y-axis ticks
    f_ax1.set_xticklabels([])  # Remove x-axis labels
    im1 = f_ax1.imshow(amp[i], vmin=np.min(amp), vmax=np.max(amp))
    f_ax1.set_title(f'{round(d[i],1)} cm', fontsize=12)

    f_ax2 = fig.add_subplot(gs[1, i])
    f_ax2.set_xticks([])    # Remove x-axis ticks
    f_ax2.set_yticks([])    # Remove y-axis ticks
    f_ax2.set_xticklabels([])  # Remove x-axis labels
    im2 = f_ax2.imshow(ph[i], vmin=np.min(ph), vmax=np.max(ph))
    if i == 0:
        f_ax1.set_ylabel('amplitude', fontsize=12)
        f_ax2.set_ylabel('phase', fontsize=12)

# Add colorbars on the right side
cbar_ax1 = fig.add_subplot(gs[0, -1])
fig.colorbar(im1, cax=cbar_ax1)

cbar_ax2 = fig.add_subplot(gs[1, -1])
fig.colorbar(im2, cax=cbar_ax2)


# Add the plots to the third row
ax3 = fig.add_subplot(gs[3, :])
std_amp = np.std(amp_images, axis=(1,2))
std_ph = np.std(ph_images, axis=(1,2))
combined = -np.std(ph_images, axis=(1,2))/np.std(amp_images, axis=(1,2))
ax3.plot(foci, unity_based_normalization(std_amp), 'b', label='std_amp')
ax3.plot(np.ones(2)*foci[np.argmin(std_amp)], np.array([0,1]), 'b--')
ax3.plot(foci, unity_based_normalization(std_ph), 'g', label='std_ph')
ax3.plot(foci, unity_based_normalization(combined), color='orange', label='combined')
ax3.plot(np.ones(2)*foci[np.argmin(combined)], np.array([0,1]), '--', color='orange')
ax3.set_xlabel('focus distance [cm]', fontsize=12)
ax3.legend()


plt.subplots_adjust(wspace=0, hspace=0)  # Adjusts the space between the subplots
plt.show()

