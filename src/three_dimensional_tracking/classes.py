# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:49:55 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import sys
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')
from  pyKoalaRemote import client
from typing import Tuple, List
import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from utilities import is_koala_running, open_koala
from scipy.ndimage import gaussian_filter, label, generate_binary_structure, center_of_mass

Image = np.ndarray
Matrix = np.ndarray

class PolynomialPlaneSubtractor:
    _image_shape = None
    _polynomial_degree = None
    _X = None
    _X_pseudoinverse = None

    @classmethod
    def get_X(cls, image_shape: Tuple[int, int], polynomial_degree: int) -> Matrix:
        """
        Get the design matrix X, calculating it if not already calculated.

        Args:
            image_shape (Tuple[int, int]): The shape of the image.
            polynomial_degree (int): The degree of the polynomial expansion.

        Returns:
            Matrix: The design matrix X.
        """
        # Input checks
        assert isinstance(image_shape, tuple) and len(image_shape) == 2, "image_shape must be a tuple of two integers"
        assert all(isinstance(dim, int) and dim >= 0 for dim in image_shape), "image_shape elements must be non-negative integers"
        assert isinstance(polynomial_degree, int) and polynomial_degree >= 0, "polynomial_degree must be a non-negative integer"

        if cls._X is None:
            cls._calculate_X(image_shape, polynomial_degree)
            cls._image_shape = image_shape
        return cls._X

    @classmethod
    def get_X_pseudoinverse(cls) -> Matrix:
        """
        Get the Moore-Penrose pseudoinverse of the design matrix X.

        Returns:
            Matrix: The Moore-Penrose pseudoinverse of X.
        """
        if cls._X_pseudoinverse is None:
            raise ValueError("X_pseudoinverse has not been calculated yet")
        return cls._X_pseudoinverse

    @classmethod
    def subtract_plane(cls, image: Image, polynomial_degree: int) -> Image:
        """
        Subtract a polynomial plane from the input image using the least squares method.

        Args:
            image (Image): The input 2D surface.
            polynomial_degree (int): The degree of the polynomial expansion.

        Returns:
            Image: The input image with the estimated polynomial plane subtracted.
        """
        # Input checks
        assert isinstance(image, np.ndarray), "image must be a numpy.ndarray"
        assert isinstance(polynomial_degree, int) and polynomial_degree >= 0, "polynomial_degree must be a non-negative integer"

        if cls._X is None or cls._X_pseudoinverse is None or cls._polynomial_degree != polynomial_degree:
            cls._calculate_X(image.shape, polynomial_degree)
        
        theta = np.dot(cls._X_pseudoinverse, image.ravel())
        plane = np.dot(cls._X, theta).reshape(image.shape)
        return image - plane

    @classmethod
    def _calculate_X(cls, image_shape: Tuple[int, int], polynomial_degree: int) -> None:
        x_coordinates, y_coordinates = np.mgrid[:image_shape[0], :image_shape[1]]
        X = np.column_stack((x_coordinates.ravel(), y_coordinates.ravel()))
        X = PolynomialFeatures(degree=polynomial_degree, include_bias=True).fit_transform(X)
        cls._image_shape = image_shape
        cls._polynomial_degree = polynomial_degree
        cls._X = X
        cls._X_pseudoinverse = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())

class Koala:
    _host = None
    _config_number = None
    _open_holo = None
    
    @classmethod
    def connect(cls, config_number: int = None):
        if not is_koala_running():
            open_koala()
        try:
            cls._host.OpenPhaseWin()
        except:
            cls._host = None
        if cls._host is None:
            host = client.pyKoalaRemoteClient()
            host.Connect('localhost')
            host.Login('admin')
            cls._host = host
        
        if cls._config_number != config_number and config_number is not None:
            cls._host.OpenConfig(config_number)
            cls._config_number = config_number
        cls._host.OpenPhaseWin()
        cls._host.OpenIntensityWin()
        cls._host.OpenHoloWin()
    
    @classmethod
    def shut_down_restart(cls):
        assert cls._host is not None, "Koala is not connected"
        cls._host.KoalaShutDown()
        time.sleep(1)
        open_koala()
        cls.connect(cls._config_number)
    
    @classmethod
    def load_hologram(cls, holo_fname):
        assert cls._host is not None, "Koala is not connected"
        assert isinstance(holo_fname, (str, Path)) and (str(holo_fname).endswith(".tif") or str(holo_fname).endswith(".TIF")), "Invalid filename"
        
        if cls._open_holo != str(holo_fname):
            cls._host.LoadHolo(str(holo_fname),1)
            cls._host.SetUnwrap2DState(True)
            cls._open_holo = str(holo_fname)
        
    @classmethod
    def set_reconstruction_distance(cls, distance):
        assert cls._host is not None, "Koala is not connected"
        assert -45 <= distance <= 45, "Number is not in the valid range [-45, 45]"

        cls._host.SetRecDistCM(distance)
        cls._host.OnDistanceChange()
      
    @classmethod
    def get_phase_image(cls):
        return cls._host.GetPhase32fImage()
    
    @classmethod
    def get_intensity_image(cls):
        return cls._host.GetIntensity32fImage()

class ImageStackBuilder:
    
    def build_stack(
        holo_fname: str,
        distances: np.ndarray,
        ph_stack: bool = True,
        amp_stack: bool = False,
        corners: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        polynomial_degree: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(holo_fname, str), "holo_fname must be a string"
        assert isinstance(distances, np.ndarray), "distances must be a numpy array"
        assert distances.size > 0, "distances must not be empty"
        assert isinstance(ph_stack, bool), "ph_stack must be a boolean"
        assert isinstance(amp_stack, bool), "amp_stack must be a boolean"
        assert isinstance(polynomial_degree, (np.integer, int)), "polynomial_degree must be an integer"
        
        koala = Koala()
        koala.load_hologram(holo_fname)
        
        ph_volume = []
        amp_volume = []
        for distance in distances:
            koala.set_reconstruction_distance(distance)
            if ph_stack:
                ph_image = koala.get_phase_image()
                ph_image = ImageStackBuilder._postprocess_image(ph_image, corners, polynomial_degree)
                ph_volume.append(ph_image)
            if amp_stack:
                amp_image = koala.get_intensity_image()
                amp_image = ImageStackBuilder._postprocess_image(amp_image, corners, polynomial_degree)
                amp_volume.append(amp_image)
        
        if ph_stack:
            ph_volume = np.stack(ph_volume, axis=0)
        if amp_stack:
            amp_volume = np.stack(amp_volume, axis=0)
        return ph_volume, amp_volume

    def _postprocess_image(image, corners, polynomial_degree):
        assert isinstance(image, np.ndarray), "image must be a numpy array"
        
        if corners is not None:
            image = image[corners[0][0]:corners[0][1],corners[1][0]:corners[1][1]]
        if polynomial_degree is not None:
            assert isinstance(polynomial_degree, (np.integer, int)), "polynomial_degree must be an integer or None"
            image = PolynomialPlaneSubtractor.subtract_plane(image, polynomial_degree)
        return image


class Object:
    
    def __init__(self, x: int, y: int, z: int, z_dist: float, mask: np.ndarray = None, image: np.ndarray = None):
        # Assert that x, y, and z have the correct data types
        assert isinstance(x, (np.integer, int)), "x must be an integer"
        assert isinstance(y, (np.integer, int)), "y must be an integer"
        assert isinstance(z, (np.integer, int)), "y must be an integer"
        assert isinstance(z_dist, (np.integer, int, np.floating, float)), "z must be a number (integer or float)"
        assert mask is None or (isinstance(mask, np.ndarray) and mask.ndim == 2), "mask needs to be a two dimensional np.ndarray or None"
        assert image is None or (isinstance(image, np.ndarray) and image.ndim == 2), "image needs to be a two dimensional np.ndarray or None"
        
        self.x = x
        self.y = y
        self.z = z
        self.z_dist = z_dist
        self.mask = mask
        self.image = image

#%%

class ObjectFinder:
    
    def __init__(self, ph_volume: np.ndarray, z_distances: np.ndarray):
        # Assert that ph_volume is a 3D numpy array and z_distances is a 1D numpy array with the correct length
        assert isinstance(ph_volume, np.ndarray) and ph_volume.ndim == 3, "ph_volume must be a 3D numpy array"
        assert isinstance(z_distances, np.ndarray) and len(z_distances) == ph_volume.shape[0], "z_distances must be a 1D numpy array with the same length as the first dimension of ph_volume"
        
        self.ph_volume: np.ndarray = ph_volume
        self.z_distances: np.ndarray = z_distances
        
    def find_objects(self, lower_threshold: float = 0.2, upper_threshld: float = 0.24, gaussian_blur_sigma: float = 1, gaussian_blur_radius: int = 2, include_images: bool = True):
        # Assert that the input parameters have the correct data types and valid values
        assert isinstance(lower_threshold, (np.integer, int, np.floating, float)) and lower_threshold >= 0, "Lower threshold must be a non-negative number"
        assert upper_threshld is None or isinstance(lower_threshold, (np.integer, int, np.floating, float)) and lower_threshold >= 0, "Upper threshold must be a non-negative number or None"
        assert gaussian_blur_sigma is None or isinstance(gaussian_blur_sigma, (np.integer, int, np.floating, float)) and gaussian_blur_sigma > 0, "Gaussian blur sigma must be a positive number or None"
        assert gaussian_blur_radius is None or isinstance(gaussian_blur_radius, (np.integer, int)) and gaussian_blur_radius >= 0, "Gaussian blur radius must be a non-negative integer or None"
        assert isinstance(include_images, bool), "include_images must be True of False"
        
        volume = self.ph_volume
        shape = volume.shape
        if gaussian_blur_sigma is not None and gaussian_blur_radius is not None:
            volume = gaussian_filter(volume, sigma=gaussian_blur_sigma, radius=gaussian_blur_radius)
        
        labeled_array, num_features = label(volume > lower_threshold, structure=generate_binary_structure(3, 3))
        objects = []
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            masked_volume = np.zeros_like(volume)
            masked_volume[mask] = volume[mask]
            if upper_threshld is not None and np.max(masked_volume) < upper_threshld:
                continue
            
            z, _, _ = np.unravel_index(masked_volume.argmax(), shape)
            y, x = center_of_mass(mask[z])
            y, x = int(np.round(y,0)), int(np.round(x,0))
            z_dist = self.z_distances[z]
            
            if include_images:
                coords = np.argwhere(mask[z])
                min_y, min_x = np.min(coords, axis=0)
                max_y, max_x = np.max(coords, axis=0)
                object_mask = mask[z, max(0,min_y-5):min(shape[1],max_y+5), max(0,min_x-5):min(shape[1],max_x+5)]
                object_image = self.ph_volume[z, max(0,min_y-5):min(shape[1],max_y+5), max(0,min_x-5):min(shape[1],max_x+5)]
                objects.append(Object(x, y, z, z_dist, object_mask.copy(), object_image.copy()))
            else:
                objects.append(Object(x, y, z, z_dist))
        return objects
    
    
#%%
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageViewer:
    def __init__(self, image_stack):
        self.image_stack = image_stack
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.img = self.ax.imshow(self.image_stack[self.current_index])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_image)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_image)

        self.connect_key_events()

    def connect_key_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.next_image(None)
        elif event.key == 'left':
            self.prev_image(None)

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.image_stack)
        self.update_image()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.image_stack)
        self.update_image()

    def update_image(self):
        self.img.set_data(self.image_stack[self.current_index])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.image_stack)}')
        self.fig.canvas.draw()

class ScatterSequence:
    def __init__(self, obj_positions_in_time):
        self.obj_positions_in_time = obj_positions_in_time
        self.current_index = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)

        positions = self.obj_positions_in_time[self.current_index]
        self.ax.scatter(positions[0], positions[1], positions[2])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.obj_positions_in_time)}')

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_image)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_image)

        self.connect_key_events()

    def connect_key_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.next_image(None)
        elif event.key == 'left':
            self.prev_image(None)

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.obj_positions_in_time)
        self.update_image()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.obj_positions_in_time)
        self.update_image()

    def update_image(self):
        positions = self.obj_positions_in_time[self.current_index]
        self.ax.scatter(positions[0], positions[1], positions[2])
        self.ax.set_title(f'Image {self.current_index + 1}/{len(self.obj_positions_in_time)}')
        self.fig.canvas.draw()

#%%
Koala.connect(281)
fname = r'D:\data\Data_Richard\2023-10-27 14-17-18\00001_00001\Holograms\00100_holo.tif'
distances = np.arange(-3,3,0.1)
ph, amp = ImageStackBuilder.build_stack(fname, distances, amp_stack=True)
#%%
objects = ObjectFinder(ph, distances).find_objects(lower_threshold = 0.27, upper_threshld  = 0.3, include_images=False)
p = ph.copy()
for obj in objects:
    p[obj.z, obj.y-3:obj.y+3, obj.x-3:obj.x+3] = np.nan
viewer = ImageViewer(p)
plt.show()
#%%
viewer = ImageViewer(ph)
plt.show()

#%%
viewer = ImageViewer((ph-np.roll(ph, 1, axis=0))[1:])
plt.show()

#%%

p = ph.copy()
for obj in objects:
    p[obj.z, obj.y-5:obj.y+5, obj.x-5:obj.x+5] = 0
viewer = ImageViewer(ph)
plt.show()
#%%
from skimage.registration import phase_cross_correlation
from scipy import ndimage

def aligned_change(ref, mov):
    shift_measured, _, __ = phase_cross_correlation(ref, mov, upsample_factor=100)
    mov = ndimage.shift(mov, shift=shift_measured, mode='constant')
    print(shift_measured)
    return ref-mov

p = np.zeros_like(ph)
for i in range(p.shape[0]-1):
    p[i] = aligned_change(ph[i], ph[i+1])
    
#%%
pp = np.zeros_like(ph)
for i in range(ph.shape[0]-1):
    pp[i] = p[i] + p[i+1]
#%%
viewer = ImageViewer(pp[:,10:790,10:790])
plt.show()
#%%
a = ph[0]
b = ph[10]
shift_measured, _, __ = phase_cross_correlation(a, b, upsample_factor=100)
b = ndimage.shift(b, shift=shift_measured, mode='constant')
plt.figure('a')
plt.imshow(a)
plt.figure('b')
plt.imshow(b)
plt.figure('bb')
plt.imshow((b-a)[20:790,20:790])
#%%
Koala.connect(281)
base_dir = r'D:\data\Data_Richard\2023-10-27 14-17-18\00001_00001\Holograms'
amp_stack = []
ph_stack = []
for i in range(10):
    fname = base_dir + os.sep + f'{str(i+200).zfill(5)}_holo.tif'
    distances = np.arange(-3,3,0.1)
    ph, amp = ImageStackBuilder.build_stack(fname, distances, amp_stack=True)
    ph_stack.append(ph)
    amp_stack.append(amp)
#%%
ph_mean = np.mean(ph_stack, axis=0)
amp_mean = np.mean(amp_stack, axis=0)
#%%
ph_mean = np.mean(ph_stack, axis=0)
#%%
viewer = ImageViewer(np.median(ph_stack, axis=0))
plt.show()
#%%












Koala.connect(280)
fname = r'D:\data\20231020_hydrogel_3D\tif_holos\2023.10.20 15-28-58_00000_TO_00010\Holograms\00009_holo.tif'
ph, _ = ImageStackBuilder.build_stack(fname, np.arange(-5,0.1,0.1))
objects = ObjectFinder(ph, np.arange(-5,0.1,0.1)).find_objects()

p = ph.copy()
for obj in objects:
    p[obj.z, obj.y-5:obj.y+5, obj.x-5:obj.x+5] = 0
viewer = ImageViewer(ph)
plt.show()

for i, ob in enumerate(objects):
    plt.figure(f'{i}_mask')
    plt.imshow(ob.mask)
    plt.show()
    plt.figure(f'{i}_image')
    plt.imshow(ob.image)
    plt.show()

#%%
Koala.connect(280)
sequence_dir = r'D:\data\20231020_hydrogel_3D\tif_holos\2023.10.20 15-28-58_00000_TO_00010\Holograms'
names = os.listdir(sequence_dir)

obj_positions_in_time = []
for name in names:
    holo_fname = sequence_dir + os.sep + name
    ph, _ = ImageStackBuilder.build_stack(holo_fname, np.arange(-5,0.1,0.1))
    objects = ObjectFinder(ph, np.arange(-5,0.1,0.1)).find_objects()
    x = [obj.x for obj in objects]
    y = [obj.y for obj in objects]
    z = [obj.z for obj in objects]
    obj_positions = (x,y,z)
    obj_positions_in_time.append(obj_positions)
    
ScatterSequence(obj_positions_in_time)