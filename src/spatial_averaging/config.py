# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:24:03 2023

@author: SWW-Bc20
"""

import os as _os
import json as _json
import numpy as _np
import subprocess as _subprocess
import time as _time
import cv2 as _cv2
import pyautogui as _pyautogui
import psutil as _psutil
import sys
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')
from pyKoalaRemote import client
from warnings import warn as _warn
from typing import Tuple, Optional

from . import __file__ as _spatavg_init  # Path to __init__ file

_SPATAVG_DIR = _os.path.dirname(_spatavg_init)
"spatial averaging lib install directory"
_LOADED = False
"Which config file was loaded"
KOALA_HOST = None
"koala host"
KOALA_CONFIG_NR = None
"selected koala configuration number"
DISPLAY_ALWAYS_ON: bool = False


focus_method: str = "combined"
optimizing_method: str = "Powell"
tolerance: Optional[float] = None

reconstruction_distance_low: float = -3.0
reconstruction_distance_high: float = -1.5
reconstruction_distance_guess: float = -2.3

plane_fit_order: int = 4
use_amp: bool = False

image_size: Tuple[int, int] = (800, 800)
px_size: float = 0.12976230680942535*1e-6
hconv: float = 794*1e-9/(2*_np.pi)
unit_code : int = 1

koala_reset_frequency: int = 10


def load_config(koala_config_nr: int = None, display_always_on: bool = False,
                focus_methodIn: str = None, optimizing_methodIn: str = None,
                reconstruction_distance_lowIn: float = None, reconstruction_distance_highIn: float = None,
                reconstruction_distance_guessIn: float = None, plane_fit_orderIn: int = None,
                koala_reset_frequencyIn: int = None):
    
    # Open Koala and load Configurations
    if not _is_koala_running():
        _open_koala()
    
    global KOALA_HOST
    try:
        KOALA_HOST.OpenPhaseWin()
    except:
        KOALA_HOST = None
    if KOALA_HOST is None:
        KOALA_HOST = client.pyKoalaRemoteClient()
        KOALA_HOST.Connect('localhost')
        KOALA_HOST.Login('admin')
    
    global KOALA_CONFIG_NR
    if koala_config_nr is not None:
        KOALA_CONFIG_NR = koala_config_nr
    if KOALA_CONFIG_NR is None:
        raise ValueError(
            """No Koala Configuration Number is selected."""
        )
    
    KOALA_HOST.OpenConfig(KOALA_CONFIG_NR)
    KOALA_HOST.OpenPhaseWin()
    KOALA_HOST.OpenIntensityWin()
    KOALA_HOST.OpenHoloWin()
    
    
    # Update the configuration settings if arguments are provided
    global DISPLAY_ALWAYS_ON
    DISPLAY_ALWAYS_ON = display_always_on
    
    if focus_methodIn is not None:
        global focus_method
        focus_method = focus_methodIn
    
    if optimizing_methodIn is not None:
        global optimizing_method
        optimizing_method = optimizing_methodIn
    
    if reconstruction_distance_lowIn is not None:
        global reconstruction_distance_low
        reconstruction_distance_low = reconstruction_distance_lowIn
    
    if reconstruction_distance_highIn is not None:
        global reconstruction_distance_high
        reconstruction_distance_high = reconstruction_distance_highIn
    
    if reconstruction_distance_guessIn is not None:
        global reconstruction_distance_guess
        reconstruction_distance_guess = reconstruction_distance_guessIn
    
    if plane_fit_orderIn is not None:
        global plane_fit_order
        plane_fit_order = plane_fit_orderIn
    
    if koala_reset_frequencyIn is not None:
        global koala_reset_frequency
        koala_reset_frequency = koala_reset_frequencyIn
    
    global _LOADED
    _LOADED = True

def set_image_variables(image_size_in: Tuple, px_size_in: float, laser_lambd: float):
    # hconv is hard coded and can not be changed, since Koala uses the wrong hconv
    global image_size
    image_size = image_size
    global px_size
    px_size = px_size_in
    global hconv
    hconv = laser_lambd/(2*_np.pi)



########################## UGLY!! Function are duplicated, to cercumvent circular import from utilities #########################
def _open_koala():
    wd = _os.getcwd()
    _os.chdir(r"C:\Program Files\LynceeTec\Koala")
    _subprocess.Popen(r"C:\Program Files\LynceeTec\Koala\Koala")
    _time.sleep(4)
    _pyautogui.typewrite('admin')
    _pyautogui.press('tab')
    _pyautogui.typewrite('admin')
    _pyautogui.press('enter')
    _time.sleep(4)
    _os.chdir(wd)
    screenshot = _pyautogui.screenshot()
    remote_log = _cv2.imread(r'spatial_averaging/images/remote_log_icon.png')
    remote_log_pos = _find_image_position(screenshot, remote_log)
    _pyautogui.click(remote_log_pos)

def _is_koala_running():
    for proc in _psutil.process_iter(['name', 'exe']):
        if proc.info['name'] == 'Koala.exe' and proc.info['exe'] == r'C:\Program Files\LynceeTec\Koala\Koala.exe':
            return True
    return False

def _find_image_position(screenshot, image, threshold=0.95):
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
    screenshot_array = _np.array(screenshot)
    image_array = _np.array(image)
    h, w = image_array.shape[:2]

    match = _cv2.matchTemplate(screenshot_array, image_array, _cv2.TM_CCOEFF_NORMED)
    # Find the position of the best match in the match matrix
    min_val, max_val, min_loc, max_loc = _cv2.minMaxLoc(match)
    if max_val<=threshold:
        return None
    
    # Get the center coordinates of the best match
    center_x = int(max_loc[0] + w/2)
    center_y = int(max_loc[1] + h/2)
    
    return center_x, center_y




