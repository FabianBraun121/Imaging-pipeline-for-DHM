# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:27:13 2023

@author: SWW-Bc20
"""
# import os
# import subprocess
# import time
# import pyautogui
# import cv2
# import numpy as np
# from pyKoalaRemote import client


# def find_image_position(screenshot, image, threshold=0.95):
#     """
#     Finds the position of a given image in a given screenshot using template matching.
#     Args:
#         screenshot: A PIL Image object of the screenshot.
#         image: A PIL Image object of the image to be located in the screenshot.
#         threshold: A float indicating the threshold above which the match is considered valid (default: 0.95).
#     Returns:
#         A tuple of two integers representing the (x, y) coordinates of the center of the image in the screenshot.
#         If the image is not found, returns None.
#     """
#     screenshot_array = np.array(screenshot)
#     image_array = np.array(image)
#     h, w = image_array.shape[:2]

#     match = cv2.matchTemplate(screenshot_array, image_array, cv2.TM_CCOEFF_NORMED)
#     # Find the position of the best match in the match matrix
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
#     if max_val<=threshold:
#         return None
    
#     # Get the center coordinates of the best match
#     center_x = int(max_loc[0] + w/2)
#     center_y = int(max_loc[1] + h/2)
    
#     return center_x, center_y


# def open_koala():
#     os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src')
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
#     pyautogui.click(find_image_position(screenshot, remote_log))

# def connect_to_remote_koala(ConfigNumber):
#     # Define KoalaRemoteClient host
#     host = client.pyKoalaRemoteClient()
#     #Ask IP address
#     IP = 'localhost'
#     # Log on Koala - default to admin/admin combo
#     host.Connect(IP)
#     host.Login('admin')
#     # Open config
#     host.OpenConfig(ConfigNumber)
#     host.OpenPhaseWin()
#     host.OpenIntensityWin()
#     host.OpenHoloWin()
#     return host

# def shut_down_restart_koala(host, ConfigNumber):
#     host.KoalaShutDown()
#     time.sleep(1)
#     open_koala()
#     host = connect_to_remote_koala(ConfigNumber)
#     return host

import subprocess
import pyautogui
# Disable the pyautogui fail-safe feature
pyautogui.FAILSAFE = False

import time

time.sleep(10)

# Define the path to the Koala application
koala_path = r"C:\Program Files\LynceeTec\Koala\Koala.exe"

# Start the Koala application
subprocess.Popen(koala_path)

# Wait for the Koala application to start
time.sleep(5)

# Define your username and password
username = "admin"
password = "admin"

# Use pyautogui to automate typing your username and password
pyautogui.typewrite(username)
pyautogui.press("tab")
pyautogui.typewrite(password)
pyautogui.press("enter")
    