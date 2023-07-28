# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:50:58 2023

@author: SWW-Bc20
"""
import os
import clr
import sys
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.ndimage
from  pyKoalaRemote import client
import cv2
    
#%%
ConfigNumber=221
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

        #%%
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

def evaluate_sobel_squared_std(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    
    # Calculate std squared sobel sharpness score
    std_sobel_sharpness_squared = np.std(grad_x ** 2 + grad_y ** 2)
    
    return -std_sobel_sharpness_squared

def evaluate_min_amp_std(amp):
    return np.std(amp)

def evaluate_combined(amp, ph):
    return -np.std(ph)/np.std(amp)


#%%
### Load this if only interested in restricted area of -3,1
save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\focus_point_evaluation_functions'
if not os.path.exists(save_path):
    os.makedirs(save_path)
x = np.zeros(400)
holos = [holo for holo in os.listdir(save_path) if holo.endswith('.tif')][:4]
#%%
downsampling = [1,2,4,8]
ph_images = np.zeros((len(holos), x.shape[0],800,800))
amp_images = np.zeros((len(holos), x.shape[0],800,800))
y_sobel_squared_std = np.zeros((len(holos), len(downsampling), x.shape[0]))
y_min_amp_std =np.zeros((len(holos), len(downsampling), x.shape[0]))
y_combined = np.zeros((len(holos), len(downsampling), x.shape[0]))
X, pseudoinverse = generate_X_and_pseudoinverse((800,800), 4)
for i in range(len(holos)):
    fname = save_path + os.sep + holos[i]
    host.LoadHolo(fname,1)
    host.SetUnwrap2DState(True)
    for j in range(x.shape[0]):
        xj = -6+j*0.02
        x[j] = xj
        host.SetRecDistCM(xj)
        host.OnDistanceChange()
        ph = host.GetPhase32fImage()
        ph = subtract_plane_X_precomputed(X, pseudoinverse, ph)
        amp = host.GetIntensity32fImage()
        amp = subtract_plane_X_precomputed(X, pseudoinverse, amp)
        ph_images[i,j] = ph
        amp_images[i,j] = amp
        for k, d in enumerate(downsampling):
            ph_down = cv2.resize(ph, (800//d, 800//d), interpolation = cv2.INTER_AREA)
            amp_down = cv2.resize(amp, (800//d, 800//d), interpolation = cv2.INTER_AREA)
            y_sobel_squared_std[i, k, j] = evaluate_sobel_squared_std(ph_down)
            y_min_amp_std[i, k, j] = evaluate_min_amp_std(amp_down)
            y_combined[i, k, j] = evaluate_combined(amp_down,ph_down)
        
    print(fname, " done")
#%%
np.save(save_path+'/x', x)
np.save(save_path+'/ph_images', ph_images)
np.save(save_path+'/amp_images', amp_images)
np.save(save_path+'/y_sobel_squared_std', y_sobel_squared_std)
np.save(save_path+'/y_min_amp_std', y_min_amp_std)
np.save(save_path+'/y_combined', y_combined)
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
ph_images = np.load(save_path+'/ph_images.npy')
amp_images = np.load(save_path+'/amp_images.npy')
y_sobel_squared_std = np.load(save_path+'/y_sobel_squared_std.npy')
y_min_amp_std = np.load(save_path+'/y_min_amp_std.npy')
y_combined = np.load(save_path+'/y_combined.npy')
#%%
fig, ax = plt.subplots(2, 2)
k = 1
for i in range(4):
    ax.flatten()[i].plot(x, min_max_normalization(y_sobel_squared_std[i,k]), 'g', label="y_sobel_squared_std")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_sobel_squared_std[i,k])], np.arange(2), 'g--')
    ax.flatten()[i].plot(x, min_max_normalization(y_min_amp_std[i,k]), 'b', label="y_min_amp_std")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_min_amp_std[i,k])], np.arange(2), 'b--')
    ax.flatten()[i].plot(x, min_max_normalization(y_combined[i,k]), 'r', label="y_combined")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_combined[i,k])], np.arange(2), 'r--')
    ax.flatten()[i].set_xlabel("lengths [cm]")
ax[0, 0].legend()
plt.show()
# plt.savefig(save_path+"/all_functions", dpi=300)
#%%
ph_images_small = np.zeros((len(holos), x.shape[0],400,400))
amp_images_small = np.zeros((len(holos), x.shape[0],400,400))
y_sobel_squared_std_small = np.zeros((len(holos), x.shape[0]))
y_min_amp_std_small =np.zeros((len(holos), x.shape[0]))
y_combined_small = np.zeros((len(holos), x.shape[0]))
X, pseudoinverse = generate_X_and_pseudoinverse((400,400), 3)
for i in range(len(holos)):
    fname = save_path + os.sep + holos[i]
    host.LoadHolo(fname,1)
    for j in range(x.shape[0]):
        xj = -3.5+j*0.01
        x[j] = xj
        host.SetRecDistCM(xj)
        host.OnDistanceChange()
        ph = host.GetPhase32fImage()[200:600,200:600]
        ph = subtract_plane_X_precomputed(X, pseudoinverse, ph)
        amp = host.GetIntensity32fImage()[200:600,200:600]
        amp = subtract_plane_X_precomputed(X, pseudoinverse, amp)
        ph_images_small[i,j] = ph
        amp_images_small[i,j] = amp
        y_sobel_squared_std_small[i, j] = evaluate_sobel_squared_std(ph)
        y_min_amp_std_small[i, j] = evaluate_min_amp_std(amp)
        y_combined_small[i, j] = evaluate_combined(amp,ph)
        
    print(fname, " done")
np.save(save_path+'/x', x)
np.save(save_path+'/ph_images_small', ph_images_small)
np.save(save_path+'/amp_images_small', amp_images_small)
np.save(save_path+'/y_sobel_squared_std_small', y_sobel_squared_std_small)
np.save(save_path+'/y_min_amp_std_small', y_min_amp_std_small)
np.save(save_path+'/y_combined_small', y_combined_small)
    
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
ph_images_small = np.load(save_path+'/ph_images_small.npy')
amp_images_small = np.load(save_path+'/amp_images_small.npy')
y_sobel_squared_std_small = np.load(save_path+'/y_sobel_squared_std_small.npy')
y_min_amp_std_small = np.load(save_path+'/y_min_amp_std_small.npy')
y_combined_small = np.load(save_path+'/y_combined_small.npy')

#%%
fig, ax = plt.subplots(2, 2)
for i in range(4):
    ax.flatten()[i].plot(x, min_max_normalization(y_sobel_squared_std_small[i]), 'g', label="y_sobel_squared_std")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_sobel_squared_std_small[i])], np.arange(2), 'g--')
    ax.flatten()[i].plot(x, min_max_normalization(y_min_amp_std_small[i]), 'b', label="y_min_amp_std")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_min_amp_std_small[i])], np.arange(2), 'b--')
    ax.flatten()[i].plot(x, min_max_normalization(y_combined_small[i]), 'r', label="y_combined")
    ax.flatten()[i].plot(np.ones(2) * x[np.argmin(y_combined_small[i])], np.arange(2), 'r--')
    ax.flatten()[i].set_xlabel("lengths [cm]")
ax[0, 0].legend()
plt.show()

#%%

from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

def interactive_image_player(frames, frame_dist=None, frame_func_eval=None):
    """This function allows you to easily display and interact with a series of images as a slider,
    with a play/pause button to animate the frames. The player also includes keyboard shortcuts for
    play/pause and frame navigation. The images can be accompanied by distance or function evaluation
    information, making it suitable for a range of applications including image analysis and video processing.

    Parameters:
    frames (numpy.ndarray): A NumPy array of image frames, with shape (num_frames, height, width, channels).
    frame_dist (list of str): A list of distances for each frame in the series. If None, the frame index is used.
    frame_func_eval (list of str): A list of function evaluations for each frame in the series.
    
    Returns:
    None
    """
    if len(frames.shape) == 3:
        num_frames, height, width = frames.shape
    else:
        num_frames, height, width, channels = frames.shape

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

    # Display the first frame
    im = ax.imshow(frames[0])

    # Create the slider axes object
    slider_ax = plt.axes([0.1, 0.15, 0.8, 0.05])
    slider = Slider(slider_ax, 'Frame', 0, num_frames-1, valinit=0)

    # Create the play/pause button axes object
    play_pause_ax = plt.axes([0.45, 0.025, 0.1, 0.1])
    play_pause_button = Button(play_pause_ax, label='▶', color='lightgoldenrodyellow', hovercolor='0.975')

    # Define the update function for the slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(frames[frame_idx])
        if frame_dist is not None:
            title = f'Dist: {np.round(frame_dist[frame_idx],2)}'
        else:
            title = f'Frame {frame_idx}'
        if frame_func_eval is not None:
            title = title + f', Func_eval: {np.around(frame_func_eval[frame_idx],3)}'
        ax.set_title(title, fontsize=16)
        fig.canvas.draw_idle()

    # Connect the slider update function to the slider object
    slider.on_changed(update)

    # Define the play/pause function
    def play_pause(event):
        nonlocal playing
        playing = not playing
        if playing:
            play_pause_button.label.set_text('❚❚')
            for i in range(int(slider.val), num_frames):
                slider.set_val(i)
                plt.pause(0.01)
                if not playing:
                    break
            if playing:
                play_pause_button.label.set_text('▶')
        else:
            play_pause_button.label.set_text('▶')
    
    # Define the key press function
    def key_press(event):
        nonlocal playing
        if event.key == ' ':
            play_pause(None)
        elif event.key == 'right':
            if slider.val < num_frames-1:
                slider.set_val(slider.val + 1)
        elif event.key == 'left':
            if slider.val > 0:
                slider.set_val(slider.val - 1)

    # Initialize the playing flag to False
    playing = False

    # Connect the play/pause function to the play/pause button object
    play_pause_button.on_clicked(play_pause)
    
    # Connect the key press function to the figure object
    fig.canvas.mpl_connect('key_press_event', key_press)

    # Set the title of the first frame
    if frame_dist is not None:
        title = f'Dist: {np.round(frame_dist[0],2)}'
    else:
        title = 'Frame 0'
    if frame_func_eval is not None:
        title = title + f', Func_eval: {np.around(frame_func_eval[0],3)}'
    ax.set_title(title, fontsize=16)
    
    # Show the plot
    plt.show()
#%%
i = 2
im = ph_images[i]
img = np.zeros((im.shape[0], 400,400))
pl = np.zeros(im.shape[0])

def _evaluate_std_ph_sobel(gray_image) -> float:
    gray_image = gray_image.clip(min=0)
    gray_image = cv2.resize(gray_image, (gray_image.shape[0]//2, gray_image.shape[1]//2), interpolation = cv2.INTER_AREA)
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    # Calculate std squared sobel sharpness score
    return np.std(grad_x ** 4 + grad_y ** 4)

def sobel_squared(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    
    # Calculate std squared sobel sharpness score
    return grad_x ** 4 + grad_y ** 4

for i in range(im.shape[0]):
    a = im[i]
    b = np.where(a>0, a, 0)
    image = cv2.resize(b, (800//2, 800//2), interpolation = cv2.INTER_AREA)
    img[i] = sobel_squared(image)
    pl[i] =_evaluate_std_ph_sobel(im[i])
#%%
interactive_image_player(img, x)
