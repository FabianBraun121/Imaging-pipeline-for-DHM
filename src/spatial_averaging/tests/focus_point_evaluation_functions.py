# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:51:04 2023

@author: SWW-Bc20
"""
"""
summary of results:
The initial idea was to measure the signal with the standard deviation. Since if
a bacteria is shown with a good resolution,

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
import cv2
    
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
#%%

        
def min_max_normalization(y):
    return (y-np.min(y))/(np.max(y)-np.min(y))

def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def evaluate_reconstruction_distance_minus_std(img):
    return -np.std(img)

def evaluate_reconstruction_distance_minus_squared_std(img):
    img *= img
    return -np.std(img)

def evaluate_entropy(img):
    marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

def evaluate_sobel(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Calculate average sobel sharpness score as the average gradient magnitude
    avg_sobel_sharpness = np.average(grad_mag)
    
    return avg_sobel_sharpness

def evaluate_sobel_squared_avg(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    
    # Calculate average squared sobel sharpness score
    avg_sobel_sharpness_squared = np.average(grad_x ** 2 + grad_y ** 2)
    
    return avg_sobel_sharpness_squared

def evaluate_sobel_squared_std(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    
    # Calculate std squared sobel sharpness score
    std_sobel_sharpness_squared = np.std(grad_x ** 2 + grad_y ** 2)
    
    return std_sobel_sharpness_squared

def evaluate_laplace_squared_std(gray_image):
    # Calculate gradient magnitude using Laplace filter
    secand_derivative = scipy.ndimage.laplace(gray_image)
    # Calculate std squared laplace sharpness score
    return np.std(secand_derivative**2)

def evaluate_fuzzy_entropy(image):
    # rescale image
    image = image*255
    
    # Calculate the gradient magnitude using Sobel filter
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)

    # Calculate the standard deviation of gradient magnitude
    gradient_std = np.std(gradient_mag)

    # Calculate fuzzy entropy
    fuzzy_entropy = 0
    for threshold in range(256):
        # Create a binary image based on threshold
        binary_image = (image > threshold).astype('float32')

        # Calculate the gradient magnitude of binary image
        gx = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Calculate the standard deviation of gradient magnitude
        binary_gradient_std = np.std(gradient_mag)

        # Calculate the membership function
        membership = np.exp(-(binary_gradient_std**2) / (gradient_std**2))

        # Add to fuzzy entropy
        if membership > 0:
            fuzzy_entropy += - membership * np.log2(membership)

    return fuzzy_entropy


#%%
### Load this if only interested in restricted area of -3,1
save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\focus_point_evaluation_functions'
if not os.path.exists(save_path):
    os.makedirs(save_path)
bounded = True
x = np.zeros(400)
#%%
### Load this if only interested in full range
save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\focus_point_evaluation_functions_full_range'
if not os.path.exists(save_path):
    os.makedirs(save_path)
bounded = False
x = np.zeros(900)
#%%
fname = save_path + r"\00000_holo.tif"
host.LoadHolo(fname,1)
images = np.zeros((x.shape[0],800,800))
y_std = np.zeros(x.shape[0])
y_squared_std = np.zeros(x.shape[0])
y_entropy = np.zeros(x.shape[0])
y_sobel = np.zeros(x.shape[0])
y_sobel_squared_avg = np.zeros(x.shape[0])
y_sobel_squared_std = np.zeros(x.shape[0])
y_laplace_squared_std = np.zeros(x.shape[0])
y_fuzzy_entropy = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    if bounded:
        xi = -3+i*0.01
    else:
        xi = -45+i*0.1
    x[i] = xi
    host.SetRecDistCM(xi)
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    img = subtract_plane(image_values, 3)
    images[i] = img
    y_std[i] = -evaluate_reconstruction_distance_minus_std(img)
    y_squared_std[i] = -evaluate_reconstruction_distance_minus_squared_std(img)
    y_entropy[i] = evaluate_entropy(img)
    y_sobel[i] = evaluate_sobel(img)
    y_sobel_squared_avg[i] = evaluate_sobel_squared_avg(img)
    y_sobel_squared_std[i] = evaluate_sobel_squared_std(img)
    y_laplace_squared_std[i] = evaluate_laplace_squared_std(img)
    y_fuzzy_entropy[i] = evaluate_fuzzy_entropy(img)
        
    print(x[i], " done")
np.save(save_path+'/x', x)
np.save(save_path+'/images', images)
np.save(save_path+'/y_std', y_std)
np.save(save_path+'/y_squared_std', y_squared_std)
np.save(save_path+'/y_entropy', y_entropy)
np.save(save_path+'/y_sobel', y_sobel)
np.save(save_path+'/y_sobel_squared_avg', y_sobel_squared_avg)
np.save(save_path+'/y_sobel_squared_std', y_sobel_squared_std)
np.save(save_path+'/y_laplace_squared_std', y_laplace_squared_std)
np.save(save_path+'/y_fuzzy_entropy', y_fuzzy_entropy)
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
images = np.load(save_path+'/images.npy')
y_std = np.load(save_path+'/y_std.npy')
y_squared_std = np.load(save_path+'/y_squared_std.npy')
y_entropy = np.load(save_path+'/y_entropy.npy')
y_sobel = np.load(save_path+'/y_sobel.npy')
y_sobel_squared_avg = np.load(save_path+'/y_sobel_squared_avg.npy')
y_sobel_squared_std = np.load(save_path+'/y_sobel_squared_std.npy')
y_laplace_squared_std = np.load(save_path+'/y_laplace_squared_std.npy')
y_fuzzy_entropy = np.load(save_path+'/y_fuzzy_entropy.npy')
#%%
plt.figure("all_functions_normalized")
plt.plot(x, min_max_normalization(y_std), label="y_std")
plt.plot(x, min_max_normalization(y_squared_std), label="y_squared_std")
plt.plot(x, min_max_normalization(y_entropy), label="y_entropy")
plt.plot(x, min_max_normalization(y_sobel), label="y_sobel")
plt.plot(x, min_max_normalization(y_sobel_squared_avg), label="y_sobel_squared_avg")
plt.plot(x, min_max_normalization(y_sobel_squared_std), label="y_sobel_squared_std")
plt.plot(x, min_max_normalization(y_laplace_squared_std), label="y_laplace_squared_std")
plt.plot(x, min_max_normalization(y_fuzzy_entropy), label="y_fuzzy_entropy")
plt.xlabel("lengths [cm]")
plt.title("std of different reconstruction lengths")
plt.legend()
plt.savefig(save_path+"/all_functions", dpi=300)
plt.show()
#%%
plt.figure("std")
plt.plot(x, y_std)
plt.xlabel("lengths [cm]")
plt.title("std of different reconstruction lengths")
plt.savefig(save_path+"/std", dpi=300)
plt.show()
#%%
plt.figure("squared std")
plt.plot(x, y_squared_std)
plt.xlabel("lengths [cm]")
plt.title("squared std of different reconstruction lengths")
plt.savefig(save_path+"/squared_std", dpi=300)
plt.show()
#%%
plt.figure("entropy")
plt.plot(x, y_entropy)
plt.xlabel("lengths [cm]")
plt.title("entropy of different reconstruction lengths")
plt.savefig(save_path+"/entropy", dpi=300)
plt.show()
#%%
plt.figure("sobel")
plt.plot(x, y_sobel)
plt.xlabel("lengths [cm]")
plt.title("sobel of different reconstruction lengths")
plt.savefig(save_path+"/y_sobel", dpi=300)
plt.show()
#%%
plt.figure("y_sobel_squared_avg")
plt.plot(x, y_sobel_squared_avg)
plt.xlabel("lengths [cm]")
plt.title("y_sobel_squared_avg of different reconstruction lengths")
plt.savefig(save_path+"/y_sobel", dpi=300)
plt.show()
#%%
plt.figure("y_sobel_squared_std")
plt.plot(x, y_sobel_squared_std)
plt.xlabel("lengths [cm]")
plt.title("y_sobel_squared_std of different reconstruction lengths")
plt.savefig(save_path+"/y_sobel_squared_std", dpi=300)
plt.show()

#%%
plt.figure("y_laplace_squared_std")
plt.plot(x, y_laplace_squared_std)
plt.xlabel("lengths [cm]")
plt.title("y_laplace_squared_std of different reconstruction lengths")
plt.savefig(save_path+"/y_laplace_squared_std", dpi=300)
plt.show()

#%%
plt.figure("fuzzy entropy")
plt.plot(x, y_fuzzy_entropy)
plt.xlabel("lengths [cm]")
plt.title("fuzzy entropy of different reconstruction lengths")
plt.savefig(save_path+"/y_fuzzy_entropy", dpi=300)
plt.show()
#%%











import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

def show_image_slider(frames, frame_labels=None):
    """Displays a series of images in the form of a NumPy array, with a play button and a slider.

    Parameters:
    frames (numpy.ndarray): A NumPy array of image frames, with shape (num_frames, height, width, channels).
    frame_labels (list of str): A list of labels for each frame in the series. If None, the frame index is used.

    Returns:
    None
    """
    num_frames, height, width= frames.shape

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Display the first frame
    im = ax.imshow(frames[0])

    # Create the slider axes object
    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, num_frames-1, valinit=0)

    # Create the play button axes object
    play_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    play_button = Button(play_ax, 'Play', color='lightgoldenrodyellow', hovercolor='0.975')

    # Define the update function for the slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(frames[frame_idx])
        if frame_labels is not None:
            ax.set_title(frame_labels[frame_idx])
        else:
            ax.set_title(str(frame_idx))
        fig.canvas.draw_idle()

    # Connect the slider update function to the slider object
    slider.on_changed(update)

    # Define the update function for the play button
    def play(event):
        for i in range(num_frames):
            slider.set_val(i)
            plt.pause(0.01)

    # Connect the play function to the play button object
    play_button.on_clicked(play)

    # Set the title of the first frame
    if frame_labels is not None:
        ax.set_title(frame_labels[0])
    else:
        ax.set_title('0')

    # Show the plot
    plt.show()
#%%
show_image_slider(images)












