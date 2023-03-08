# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:45:14 2023

@author: SWW-Bc20
"""
import sys
import clr
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')
# Import KoalaRemoteClient
clr.AddReference("LynceeTec.KoalaRemote.Client")
from LynceeTec.KoalaRemote.Client import KoalaRemoteClient
import skimage.restoration as skir
import numpy.ma as ma
import numpy as np
from PyQt5.QtWidgets import QFileDialog

def connect_to_remote_koala(ConfigNumber):
    # Define KoalaRemoteClient host
    host=KoalaRemoteClient()
    #Ask IP address
    IP = 'localhost'
    # Log on Koala - default to admin/admin combo
    username = 'admin'
    [ret,username] = host.Connect(IP,username,True);
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