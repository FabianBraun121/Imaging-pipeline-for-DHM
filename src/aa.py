# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:42:59 2023

@author: SWW-Bc20
"""
from pyvirtualdisplay.smartdisplay import SmartDisplay
import os
import pyautogui
import time
import subprocess
import cv2
import numpy as np

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
    remote_log = cv2.imread(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging\imagesremote_log_icon.png')
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

#%%
display = SmartDisplay(visible=1, size=(1600, 900))
#%%
display.start()
os.environ['DISPLAY'] = ':0'

#%%
open_koala()


#display.stop()
#%%
import os
os.environ['DISPLAY'] = 'localhost:0.0'