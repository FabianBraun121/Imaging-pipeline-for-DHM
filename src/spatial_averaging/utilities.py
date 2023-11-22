# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:45:14 2023

@author: SWW-Bc20
"""
import os
import sys
import numpy as np
import time
import numpy.typing as npt
import cv2
import subprocess
import pyautogui
import psutil
from typing import Tuple
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from skimage.registration import phase_cross_correlation
from PyQt5.QtWidgets import QFileDialog
import ctypes
import skimage.transform as trans
from scipy import ndimage
from  pyKoalaRemote import client
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')

Image = npt.NDArray[np.float32]
CplxImage = npt.NDArray[np.complex64]
Mask = npt.NDArray[np.uint8]
Matrix = np.ndarray

def gradient_squared(image: Image) -> Image:
    grad_x = ndimage.sobel(image, axis=0)
    grad_y = ndimage.sobel(image, axis=1)
    return (grad_x**2+grad_y**2)

def grid_search_2d(
    image1: Image, image2: Image, x_mid: float, y_mid: float,
    x_length: float, y_length: float, local_searches: int) -> (float, float):
    # Initialize the initial grid boundaries.
    x_start, x_end = x_mid - x_length/2, x_mid + x_length/2
    y_start, y_end = y_mid - y_length/2, y_mid + y_length/2
    
    count = 0
    while count in range(local_searches):
        # Create a grid based on the current boundaries.
        x_values = np.linspace(x_start, x_end, 5)
        y_values = np.linspace(y_start, y_end, 5)
        
        # Initialize variables to track the minimum and its location.
        min_value = float('inf')
        min_x, min_y = None, None
        
        # Evaluate the function at each point in the grid.
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):
                if (i+j)%2==0:
                    value = evaluate_phase_shift_error(image1, image2, x, y)
                    if value < min_value:
                        min_value = value
                        min_x, min_y = x, y
        
        # Check if the minimum is at the edge or in the middle.
        if (
            min_x == x_start or min_x == x_end or
            min_y == y_start or min_y == y_end
        ):
            # If the minimum is at the edge, expand the search space.
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
        else:
            count += 1
            # If the minimum is in the middle, reduce the grid size.
            x_length /= 3
            y_length /= 3
            x_start, x_end = min_x - x_length/2, min_x + x_length/2
            y_start, y_end = min_y - y_length/2, min_y + y_length/2
    return min_x, min_y

def evaluate_phase_shift_error(image1: Image, image2: Image, rotation: float, zoomlevel: float) -> float:
    im = trans.rotate(image2, rotation, mode="edge")
    im = zoom(im, zoomlevel)
    try:
        shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10, normalization=None)
    except:
        shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10)
    return error

def zoom(I: Image, zoomlevel: float) -> Image:
    if zoomlevel <= 0:
        return np.zeros_like(I)
    oldshape = I.shape
    I_zoomed = np.zeros_like(I)
    I = trans.rescale(I, zoomlevel, mode="edge")
    if zoomlevel<1:
        i0 = (
            round(oldshape[0]/2 - I.shape[0]/2),
            round(oldshape[1]/2 - I.shape[1]/2),
        )
        I_zoomed[i0[0]:i0[0]+I.shape[0], i0[1]:i0[1]+I.shape[1]] = I
        return I_zoomed
    else:
        I = trans.rescale(I, zoomlevel, mode="edge")
        i0 = (
            round(I.shape[0] / 2 - oldshape[0] / 2),
            round(I.shape[1] / 2 - oldshape[1] / 2),
        )
        I = I[i0[0] : (i0[0] + oldshape[0]), i0[1] : (i0[1] + oldshape[1])]
        return I
    

def Open_Directory(directory, message):
    fname = QFileDialog.getExistingDirectory(None, message, directory, QFileDialog.ShowDirsOnly)
    return fname


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
    remote_log = cv2.imread(r'spatial_averaging/images/remote_log_icon.png')
    remote_log_pos = find_image_position(screenshot, remote_log)
    pyautogui.click(remote_log_pos)

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

def is_screen_active():
    user32 = ctypes.windll.user32
    return user32.GetForegroundWindow() != 0
    
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
        
        # shortest time would be 1826, assumed that restarting takes 10 seconds and the time increase is 0.006ms per evaluation.
        # parameters have been experimentally shown. Minimum calculated with sqrt((2*reset_time)/time_increase)
        if cls._evaluation_counter >= 5000 and is_screen_active():
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


class ValueTracker:
    def __init__(self):
        self.value = None
        self.value_list = []
        self.average_at = 10
    
    def append_value(self, value):
        self.value_list.append(value)
        if len(self.value_list) == self.average_at:
            self.calculate_average()
    
    def calculate_average(self):
        pass  # Initialize in subclass