# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:04:26 2023

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
from scipy.ndimage import gaussian_filter, label, find_objects, generate_binary_structure, center_of_mass
from skimage.measure import regionprops
import pandas as pd
import scipy.io
import pyautogui
import subprocess
import cv2
import psutil



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

def find_image_position(screenshot, image, threshold=0.95):
    """
    Finds the position of a given image in a given screenshot using template matching.
    Args:
        screenshot: A PIL Image object of the screenshot.
        image: A PIL Image object of the image to be located in the screenshot.
        threshold: A float indicating the threshold above which the match is considered valid (default: 0.95).
    Returns:
        A tuple of two integers representing the (x, y) coordinates of the center of the image in the screenshot.
        If the image is not found, returns None.
    """
    screenshot_array = np.array(screenshot)
    image_array = np.array(image)
    h, w = image_array.shape[:2]

    match = cv2.matchTemplate(screenshot_array, image_array, cv2.TM_CCOEFF_NORMED)
    # Find the position of the best match in the match matrix
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if max_val<=threshold:
        return None
    
    # Get the center coordinates of the best match
    center_x = int(max_loc[0] + w/2)
    center_y = int(max_loc[1] + h/2)
    
    return center_x, center_y

def is_koala_running():
    for proc in psutil.process_iter(['name', 'exe']):
        if proc.info['name'] == 'Koala.exe' and proc.info['exe'] == r'C:\Program Files\LynceeTec\Koala\Koala.exe':
            return True
    return False

def open_koala():
    wd = os.getcwd()
    os.chdir(r"C:\Program Files\LynceeTec\Koala")
    subprocess.Popen(r"C:\Program Files\LynceeTec\Koala\Koala")
    time.sleep(4)
    pyautogui.typewrite('admin')
    pyautogui.press('tab')
    pyautogui.typewrite('admin')
    pyautogui.press('enter')
    time.sleep(4)
    os.chdir(wd)
    screenshot = pyautogui.screenshot()
    remote_log = cv2.imread(r'remote_log_icon.png')
    remote_log_pos = find_image_position(screenshot, remote_log)
    pyautogui.click(remote_log_pos)

class Koala:
    _host = None
    _config_number = None
    _open_holo = None
    _evaluation_counter = 0
    
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
        
        if config_number is not None:
            cls._host.OpenConfig(config_number)
            cls._config_number = config_number
        cls._host.OpenPhaseWin()
        cls._host.OpenIntensityWin()
        cls._host.OpenHoloWin()
    
    @classmethod
    def shut_down_restart(cls):
        assert cls._host is not None, "Koala is not connected"
        cls._host.KoalaShutDown()
        cls._host = None
        cls._open_holo = None
        cls._evaluation_counter = 0
        time.sleep(1)
        open_koala()
        cls.connect(cls._config_number)
    
    @classmethod
    def load_hologram(cls, holo_fname):
        assert cls._host is not None, "Koala is not connected"
        assert isinstance(holo_fname, (str, Path)) and (str(holo_fname).endswith(".tif") or str(holo_fname).endswith(".TIF")), "Invalid filename"
        
        if cls._evaluation_counter >= 3600:
            cls.shut_down_restart()
        
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
        cls._evaluation_counter += 1
      
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

class Bacteria:
    
    def __init__(self, x: int, y: int, z: int, z_dist: float, mask: np.ndarray = None, area: int = None, circularity: float = None, mean_ph_dif: float = None):
        # Assert that x, y, and z have the correct data types
        assert isinstance(x, (np.integer, int)), "x must be an integer"
        assert isinstance(y, (np.integer, int)), "y must be an integer"
        assert isinstance(z, (np.integer, int)), "y must be an integer"
        assert isinstance(z_dist, (np.integer, int, np.floating, float)), "z must be a number (integer or float)"
        
        self.x = x
        self.y = y
        self.z = z
        self.z_dist = z_dist
        self.mask = mask
        self.area = area
        self.circularity = circularity
        self.mean_ph_dif = mean_ph_dif


class BacteriaFinder:
    
    def __init__(self, ph_volume: np.ndarray, z_distances: np.ndarray):
        # Assert that ph_volume is a 3D numpy array and z_distances is a 1D numpy array with the correct length
        assert isinstance(ph_volume, np.ndarray) and ph_volume.ndim == 3, "ph_volume must be a 3D numpy array"
        assert isinstance(z_distances, np.ndarray) and len(z_distances) == ph_volume.shape[0], "z_distances must be a 1D numpy array with the same length as the first dimension of ph_volume"
        
        self.ph_volume: np.ndarray = ph_volume
        self.z_distances: np.ndarray = z_distances
        
    def find_bacteria(self, lower_threshold: float = 0.15, upper_threshld: float = 0.24, gaussian_blur_sigma: float = 1, gaussian_blur_radius: int = 2) -> List[Bacteria]:
        # Assert that the input parameters have the correct data types and valid values
        assert isinstance(lower_threshold, (np.integer, int, np.floating, float)) and lower_threshold >= 0, "Lower threshold must be a non-negative number"
        assert upper_threshld is None or isinstance(lower_threshold, (np.integer, int, np.floating, float)) and lower_threshold >= 0, "Upper threshold must be a non-negative number or None"
        assert gaussian_blur_sigma is None or isinstance(gaussian_blur_sigma, (np.integer, int, np.floating, float)) and gaussian_blur_sigma > 0, "Gaussian blur sigma must be a positive number or None"
        assert gaussian_blur_radius is None or isinstance(gaussian_blur_radius, (np.integer, int)) and gaussian_blur_radius >= 0, "Gaussian blur radius must be a non-negative integer or None"
        
        volume = self.ph_volume.copy()
        if gaussian_blur_sigma is not None and gaussian_blur_radius is not None:
            volume = gaussian_filter(volume, sigma=gaussian_blur_sigma, radius=gaussian_blur_radius)
        
        labeled_array, num_features = label(volume > lower_threshold, structure=generate_binary_structure(3, 3))
        object_slices = find_objects(labeled_array)
        bacterias = []
        
        for i, object_slice in enumerate(object_slices, start=1):
            mask = labeled_array[object_slice] == i
            masked_volume = volume[object_slice]
            masked_volume[~mask] = 0
            max_value = np.max(masked_volume)
            if upper_threshld is not None and max_value < upper_threshld:
                continue
            # object is contantly there in the z direction: is probably due to unclean image
            if 2<np.ptp(self.z_distances[object_slice[0]]):
                continue
            
            zs, _, _ = np.unravel_index(masked_volume.argmax(), masked_volume.shape)
            ys, xs = center_of_mass(mask[zs])
            ys, xs = int(np.round(ys,0)), int(np.round(xs,0))
            z, y, x = zs+object_slice[0].start, ys+object_slice[1].start, xs+object_slice[2].start
            z_dist = self.z_distances[z]
            
            prop = regionprops(mask[zs].astype(np.uint8))[0]
            
            area = prop.area
            circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
            mean_ph_dif = np.mean(self.ph_volume[object_slice][mask])
            
            bacterias.append(Bacteria(x, y, z, z_dist, mask[zs], area, circularity, mean_ph_dif))
        return bacterias
    


Koala.connect(281)
base_dir = r'D:\data\Data_Richard\2023-10-27 14-17-18\00001_00001\Holograms'
fnames = os.listdir(base_dir)
distances = np.arange(-3,3,0.1)
start_time = time.time()

data = []
for fname in fnames:
    timestep = int(fname[:5])
    ph, _ = ImageStackBuilder.build_stack(base_dir + os.sep + fname, distances)
    bacterias = BacteriaFinder(ph, distances).find_bacteria(lower_threshold = 0.17, upper_threshld  = 0.23, gaussian_blur_sigma=1, gaussian_blur_radius=3)
    for bacteria in bacterias:
        row = {key: value for key, value in vars(bacteria).items() if key != 'mask'}
        row['timestep'] = timestep
        data.append(row)

df = pd.DataFrame(data)
end_time = time.time()
duration = np.round(end_time-start_time,0)
print(f'It took {duration//60} min {duration%60} s, {np.round(duration/len(fnames),1)} s per image')

data_dict = {'bacterias': df.to_dict(orient='list')}
scipy.io.savemat(r'D:\data\Data_Richard\2023-10-27 14-17-18\bacteria_data.mat', data_dict)
df.to_csv(r'D:\data\Data_Richard\2023-10-27 14-17-18\bacteria_data.csv', index=False)
