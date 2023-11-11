# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:45:14 2023

@author: SWW-Bc20
"""
import sys
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')
import os
from  pyKoalaRemote import client
import skimage.restoration as skir
import numpy.ma as ma
import numpy as np
import numpy.typing as npt
import subprocess
import time
import pyautogui
import cv2
import psutil
import skimage.transform as trans
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from PyQt5.QtWidgets import QFileDialog
from typing import Tuple, List

sys.path.append("..")
import config as cfg

Image = npt.NDArray[np.float32]
Matrix = np.ndarray

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

def crop_image(image_array, crop_coords):
    # Extract the crop coordinates
    ymin, ymax = crop_coords[0][0], crop_coords[0][1]
    xmin, xmax = crop_coords[1][0], crop_coords[1][1]
    
    return image_array[ymin:ymax, xmin:xmax]

def evaluate_phase_shift_error(image1: Image, image2: Image, rotation: float, zoomlevel: float) -> float:
    im = trans.rotate(image2, rotation, mode="edge")
    im = zoom(im, zoomlevel)
    try:
        shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10, normalization=None)
    except:
        shift_measured, error, phasediff = phase_cross_correlation(image1, im, upsample_factor=10)
    return error

def Open_Directory(directory, message):
    fname = QFileDialog.getExistingDirectory(None, message, directory, QFileDialog.ShowDirsOnly)
    return fname

def get_result_unwrap(phase, mask=None):
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))
    
def get_masks_corners(mask):
    non_zero_indices = np.nonzero(mask)
    ymin, ymax = int(np.min(non_zero_indices[0])), int(np.max(non_zero_indices[0])+1)
    xmin, xmax = int(np.min(non_zero_indices[1])), int(np.max(non_zero_indices[1])+1)
    return ((ymin, ymax), (xmin, xmax))

def is_koala_running():
    for proc in psutil.process_iter(['name', 'exe']):
        if proc.info['name'] == 'Koala.exe' and proc.info['exe'] == r'C:\Program Files\LynceeTec\Koala\Koala.exe':
            return True
    return False
    
def logout_login_koala():
    cfg.KOALA_HOST.Logout()
    time.sleep(0.1)
    cfg.KOALA_HOST.Connect('localhost')
    cfg.KOALA_HOST.Login('admin')
    cfg.KOALA_HOST.OpenConfig(cfg.koala_config_nr)
    cfg.KOALA_HOST.OpenPhaseWin()
    cfg.KOALA_HOST.OpenIntensityWin()
    cfg.KOALA_HOST.OpenHoloWin()

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

def shut_down_restart_koala():
    cfg.KOALA_HOST.KoalaShutDown()
    time.sleep(1)
    open_koala()
    cfg.KOALA_HOST = connect_to_remote_koala(cfg.koala_config_nr)
    
def start_koala():
    # Open Koala and load Configurations
    if not is_koala_running():
        open_koala()
    
    try:
        cfg.KOALA_HOST.OpenPhaseWin()
    except:
        cfg.KOALA_HOST = None
    if cfg.KOALA_HOST is None:
        cfg.KOALA_HOST = client.pyKoalaRemoteClient()
        cfg.KOALA_HOST.Connect('localhost')
        cfg.KOALA_HOST.Login('admin')
    
    cfg.KOALA_HOST.OpenConfig(cfg.koala_config_nr)
    cfg.KOALA_HOST.OpenPhaseWin()
    cfg.KOALA_HOST.OpenIntensityWin()
    cfg.KOALA_HOST.OpenHoloWin()

def zoom(I: Image, zoomlevel: float) -> Image:
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
    