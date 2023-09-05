# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:33:35 2023

@author: SWW-Bc20
"""
import os
import clr
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.ndimage
from  pyKoalaRemote import client
import cv2
from scipy import ndimage

def min_max_normalization(y):
    return (y-np.min(y))/(np.max(y)-np.min(y))

def generate_X_and_pseudoinverse(field_shape, plane_degree):
    X1, X2 = np.mgrid[:field_shape[0], :field_shape[1]]
    X = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=True).fit_transform(X)
    pseudoinverse = np.dot( np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
    return X, pseudoinverse

def subtract_plane_X_precomputed(X, pseudoinverse, field):
    theta = np.dot(pseudoinverse, field.reshape(-1))
    plane = np.dot(X, theta).reshape(field.shape[0], field.shape[1])
    return field-plane

def sharpness(image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    # Calculate std squared sobel sharpness score
    return np.std(grad_x ** 4 + grad_y ** 4)

def std_amp(image):
    return np.std(image)

#%%
ConfigNumber=219
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


save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images\data\focus_distances_run'
fname_holo = r'F:\C11_20230217\2023-02-17 11-13-34\00001\00001_00001\Holograms\00000_holo.tif'
#%%
x = np.zeros(400)
ph_images = np.zeros((x.shape[0],800,800))
amp_images = np.zeros((x.shape[0],800,800))
y_sharpness = np.zeros((x.shape[0]))
y_std_amp = np.zeros((x.shape[0]))
X, pseudoinverse = generate_X_and_pseudoinverse((800,800), 4)

host.LoadHolo(fname_holo,1)
host.SetUnwrap2DState(True)
for i in range(x.shape[0]):
    xi = -3.5+i*0.01
    x[i] = xi
    host.SetRecDistCM(xi)
    host.OnDistanceChange()
    ph = host.GetPhase32fImage()
    ph = subtract_plane_X_precomputed(X, pseudoinverse, ph)
    amp = host.GetIntensity32fImage()
    amp = subtract_plane_X_precomputed(X, pseudoinverse, amp)
    ph_images[i] = ph
    amp_images[i] = amp
    y_sharpness[i] = sharpness(ph)
    y_std_amp[i] = std_amp(amp)

#%%
plt.plot(x, min_max_normalization(y_sharpness))
plt.plot(x, min_max_normalization(y_std_amp))

#%%
def display_frames_with_plot(amp, ph, x, y_std_amp, y_sharpness):
    current_index = 0
    running = True
    lw = 2

    # Create the figure and subplots
    fig, (ax_amp, ax_ph, ax_plot) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set a common colorbar for all images
    amp_norm = plt.Normalize(vmin=amp.min(), vmax=amp.max())
    ph_norm = plt.Normalize(vmin=ph.min(), vmax=ph.max())
    cmap = plt.cm.viridis
    
    # Initial display
    ax_amp.imshow(amp[current_index], cmap=cmap, norm=amp_norm)
    ax_amp.set_xticks([])
    ax_amp.set_yticks([])
    ax_amp.set_title("")

    ax_ph.imshow(ph[current_index], cmap=cmap, norm=ph_norm)
    ax_ph.set_xticks([])
    ax_ph.set_yticks([])
    ax_ph.set_title("")
    
    ax_amp.spines['top'].set_linewidth(lw)
    ax_amp.spines['bottom'].set_linewidth(lw)
    ax_amp.spines['left'].set_linewidth(lw)
    ax_amp.spines['right'].set_linewidth(lw)
    ax_ph.spines['top'].set_linewidth(lw)
    ax_ph.spines['bottom'].set_linewidth(lw)
    ax_ph.spines['left'].set_linewidth(lw)
    ax_ph.spines['right'].set_linewidth(lw)
    
    ax_plot.plot(x, y_std_amp, label='std amplitude', color='b')
    ax_plot.plot(x, y_std_amp, label='phase sharpness', color='g')
    ax_plot.plot(x[current_index]*np.arange(2), np.arange(2), label='current distance', color='orange')
    ax_plot.set_xlabel('Reconstruction distance [cm]', fontsize=14)
    ax_plot.set_ylabel('Normalized function output', fontsize=14)
    ax_plot.legend()

    def update_display():
        nonlocal current_index
        
        # Clear previous images/lines
        ax_amp.cla()
        ax_amp.imshow(amp[current_index], cmap=cmap, norm=amp_norm)
        ax_amp.set_xticks([])
        ax_amp.set_yticks([])
        ax_amp.set_title("Amplitude")
        ax_ph.cla()
        ax_ph.imshow(ph[current_index], cmap=cmap, norm=ph_norm)
        ax_ph.set_xticks([])
        ax_ph.set_yticks([])
        ax_ph.set_title("Phase")
        ax_amp.spines['top'].set_linewidth(lw)
        ax_amp.spines['bottom'].set_linewidth(lw)
        ax_amp.spines['left'].set_linewidth(lw)
        ax_amp.spines['right'].set_linewidth(lw)
        ax_ph.spines['top'].set_linewidth(lw)
        ax_ph.spines['bottom'].set_linewidth(lw)
        ax_ph.spines['left'].set_linewidth(lw)
        ax_ph.spines['right'].set_linewidth(lw)
        
        ax_plot.cla()
        ax_plot.plot(x, y_std_amp, label='std amplitude', color='b')
        ax_plot.plot(x, y_sharpness, label='phase sharpness', color='g')
        ax_plot.plot(x[current_index]*np.ones(2), np.arange(2), label='current distance', color='orange')
        ax_plot.set_xlabel('Reconstruction distance [cm]', fontsize=12)
        ax_plot.set_ylabel('Normalized function output', fontsize=12)
        ax_plot.tick_params(axis='both', labelsize=10)
        ax_plot.legend(fontsize=10)
        
        fig.canvas.draw()
    
    def endscreen():
        x_min = np.argmin(y_std_amp[:y_std_amp.shape[0]//2])
        # Clear previous images/lines
        ax_amp.cla()
        ax_amp.imshow(amp[x_min], cmap=cmap, norm=amp_norm)
        ax_amp.set_xticks([])
        ax_amp.set_yticks([])
        ax_amp.set_title("Amplitude")
        ax_ph.cla()
        ax_ph.imshow(ph[x_min], cmap=cmap, norm=ph_norm)
        ax_ph.set_xticks([])
        ax_ph.set_yticks([])
        ax_ph.set_title("Phase")
        ax_amp.spines['top'].set_linewidth(lw)
        ax_amp.spines['bottom'].set_linewidth(lw)
        ax_amp.spines['left'].set_linewidth(lw)
        ax_amp.spines['right'].set_linewidth(lw)
        ax_ph.spines['top'].set_linewidth(lw)
        ax_ph.spines['bottom'].set_linewidth(lw)
        ax_ph.spines['left'].set_linewidth(lw)
        ax_ph.spines['right'].set_linewidth(lw)
        
        ax_plot.cla()
        ax_plot.plot(x, y_std_amp, label='std amplitude', color='b')
        ax_plot.plot(x, y_sharpness, label='phase sharpness', color='g')
        ax_plot.plot(x[x_min]*np.ones(2), np.arange(2), label='current distance', color='orange')
        ax_plot.set_xlabel('Reconstruction distance [cm]', fontsize=12)
        ax_plot.set_ylabel('Normalized function output', fontsize=12)
        ax_plot.tick_params(axis='both', labelsize=10)
        ax_plot.legend(fontsize=10)
        
        fig.canvas.draw()


    def on_key_press(event):
        nonlocal current_index, running

        if event.key == 'right' and current_index < len(x)-1:
            current_index += 1
            update_display()
        elif event.key == 'left' and current_index > 0:
            current_index -= 1
            update_display()
        elif event.key in ['k', 'space']:
            running = not running

    def automatic_update(frame):
        nonlocal current_index, running
        if running:
            if current_index < len(x) - 1:
                current_index += 1
                update_display()
            else:
                running = False  # Stop the animation when reaching the end
                endscreen()
                ani.event_source.stop()  # Stop the animation loop
        return []
    
    def on_close(event):
        # This function is called when the plot window is closed
        ani.event_source.stop()  # Stop the animation loop
        plt.close()  # Close the plot window
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('close_event', on_close) 

    update_display()  # Display the initial frame before animation starts
    ani = animation.FuncAnimation(fig, automatic_update, interval=40, save_count=len(x)+1)
    plt.show()
    
    # Save animation as a video
    ani.save(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images\data\focus_distances_run\fucus_run.mp4', writer='ffmpeg')  # Uncomment this line to save the animation as a video
    

display_frames_with_plot(amp_images, ph_images, x, min_max_normalization(y_std_amp), min_max_normalization(y_sharpness))