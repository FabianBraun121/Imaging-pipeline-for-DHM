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
import subprocess
import time
import pyautogui
from mss import mss
import psutil
import win32api
import win32con
import cv2
from PyQt5.QtWidgets import QFileDialog

from . import config as cfg


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

def Open_Directory(directory, message):
    #print(directory)
    fname = QFileDialog.getExistingDirectory(None, message, directory, QFileDialog.ShowDirsOnly)
#    if python_vers == "3.x":
#        fname = fname[0]
    return fname

def get_result_unwrap(phase, mask=None):
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))
    
def logout_login_koala():
    cfg.KOALA_HOST.Logout()
    time.sleep(0.1)
    cfg.KOALA_HOST.Connect('localhost')
    cfg.KOALA_HOST.Login('admin')
    cfg.KOALA_HOST.OpenConfig(cfg.KOALA_CONFIG_NR)
    cfg.KOALA_HOST.OpenPhaseWin()
    cfg.KOALA_HOST.OpenIntensityWin()
    cfg.KOALA_HOST.OpenHoloWin()

def open_koala():
    wd = os.getcwd()
    os.chdir(r"C:\Program Files\LynceeTec\Koala")
    subprocess.Popen(r"C:\Program Files\LynceeTec\Koala\Koala")
    time.sleep(3)
    # Use the subprocess module to spawn a new process and communicate with it
    p = subprocess.Popen(["powershell", "-Command", "$wshell = New-Object -ComObject wscript.shell;$wshell.SendKeys('admin{TAB}admin{ENTER}')"])
    p.wait()
    time.sleep(2)
    os.chdir(wd)
    remote_log = cv2.imread(cfg._SPATAVG_DIR + os.sep + r'images/remote_log_icon.png')
    # Use the mss library to capture a screenshot
    with mss() as sct:
        screenshot = sct.shot()
    x, y = find_image_position(cv2.imread(screenshot), remote_log)
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

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

def shut_down_restart_koala():
    cfg.KOALA_HOST.KoalaShutDown()
    time.sleep(1)
    open_koala()
    cfg.KOALA_HOST = connect_to_remote_koala(cfg.KOALA_CONFIG_NR)
