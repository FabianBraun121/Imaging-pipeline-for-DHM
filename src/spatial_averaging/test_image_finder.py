# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:58:08 2023

@author: SWW-Bc20
"""
import os
import sys
import numpy as np
import pygetwindow as gw
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
import matplotlib.pyplot as plt
#%%

def find_image_position(screenshot, image, threshold=0.8):
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

# def open_koala():
#     wd = os.getcwd()
#     os.chdir(r"C:\Program Files\LynceeTec\Koala")
#     subprocess.Popen(r"C:\Program Files\LynceeTec\Koala\Koala")
#     time.sleep(4)
#     pyautogui.typewrite('admin')
#     pyautogui.press('tab')
#     pyautogui.typewrite('admin')
#     pyautogui.press('enter')
#     time.sleep(4)
#     os.chdir(wd)
#     screenshot = pyautogui.screenshot()
#     remote_log = cv2.imread(r'spatial_averaging/images/remote_log_icon.png')
#     remote_log_pos = find_image_position(screenshot, remote_log)
#     pyautogui.click(remote_log_pos)


def koala_to_the_front():
    while True:
        process = gw.getWindowsWithTitle("Koala")
        if len(process) != 0:
            if process:
                process[0].activate()
                process[0].activate()
                time.sleep(0.1)
                return
        time.sleep(0.1)

def open_koala():
    user_identification_image = cv2.imread(r'spatial_averaging/images/user_identification.png')
    remote_log_icon = cv2.imread(r'spatial_averaging/images/remote_log_icon.png')
    
    wd = os.getcwd()
    os.chdir(r"C:\Program Files\LynceeTec\Koala")
    subprocess.Popen(r"C:\Program Files\LynceeTec\Koala\Koala")
    
    id_pos = None
    while id_pos is None:
        koala_to_the_front()
        screenshot = pyautogui.screenshot()
        id_pos = find_image_position(screenshot, user_identification_image, threshold=0.45)
    
    # Type credentials
    koala_to_the_front()
    pyautogui.click(id_pos, interval=0.0)
    pyautogui.typewrite('admin', interval=0.0)
    pyautogui.press('tab', interval=0.0)
    pyautogui.typewrite('admin', interval=0.0)
    pyautogui.press('enter', interval=0.0)
    
    os.chdir(wd)
    
    
def connect_to_remote_koala():
    remote_log_icon = cv2.imread(r'spatial_averaging/images/remote_log_icon.png')
    host = client.pyKoalaRemoteClient()
    connected = host.Connect('localhost')
    while not connected:
        koala_to_the_front()
        screenshot = pyautogui.screenshot()
        log_icon_pos = find_image_position(screenshot, remote_log_icon)
        pyautogui.click(log_icon_pos, interval=0.0)
        connected = host.Connect('localhost')
    host.Login('admin')
    return host

# Call the open_koala function
open_koala()

#%%
host = client.pyKoalaRemoteClient()
host.Connect('localhost')
#%%
remote_message_log = cv2.imread(r'spatial_averaging/images/remote_message_log.png')
while True:
    screenshot = pyautogui.screenshot()
    message_log_pos = find_image_position(screenshot, remote_message_log, threshold=0.6)
    time.sleep(0.1)
