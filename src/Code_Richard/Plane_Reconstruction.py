# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:19:48 2022

Script to reconstruct planes of a hologram sequence
Requires Koala to be open and production tab (remote) tab to be open

@author: henshaw
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from tkinter import Tk,filedialog
import fnmatch
import pathlib
import clr
import glob
import shutil
# FastSplitter extra requirements
import PIL.Image
import csv
import FastSplitter

###################  User Settings ############################################

# Settings
ExpDate = '221020'
ExpName = 'Exp10'
ConfigNumber = 212 # Configuration ID 
HoloMethod = 'Fast' #  'Slow' or 'Fast' saving (of initial experiment)
NfastFrames = 1000 # If a fast video, how many total frames
inputdirectory = Path(pathlib.PureWindowsPath(r'G:\\').joinpath('DHM',ExpDate,ExpName))
FrameNums = [(0,250)] # Frames to do the reconstruction over
SaveMethod = 1 # 0 = bin only, 1 = tif only, 2 = both ONLY TIF CURRENTLY WORKING

#Directories
outputMainDirectory = Path(pathlib.PureWindowsPath(r'D:\\').joinpath('DHM_Reconstructions',ExpDate,ExpName))
outputMainDirectory.mkdir(parents=True,exist_ok=True)
SplittingDirectory = Path(pathlib.PureWindowsPath(r'G:\\').joinpath('DHM',ExpDate,'SplitHolograms'))
SplittingDirectory.mkdir(parents=True,exist_ok=True)

# Reconstruction planes / heights
recDistLeft = -3 # Left most reconstruction distance
recDistRight = 1 # Right most reconstruction distance
zstep = 0.05 # Reconstruction height step

# Phase correction fitting - integer inputs - NOT IMPLEMENTED YET
# degreeOfFit = 3
# if degreeOfFit > 0:
#     fitMethod = 1 # Polynomial fit method
# else:
#     fitMethod = 0 # Tilt method 
#     degreeOfFit = 1

###################  Saving Directory #########################################

# Check how many reconstuctions have been done, make new folder for new reconstruction
recon_folders = outputMainDirectory.glob('Reconstruction_*')
recon_num = 0
for i in recon_folders:
    recon_num = recon_num + 1
recon_num = recon_num + 1
outputdirectory = Path(pathlib.PureWindowsPath(outputMainDirectory).joinpath('Reconstruction_' + str(recon_num).zfill(3)))
outputdirectory.mkdir(parents=True,exist_ok=True)

####################  Connect to Koala  #######################################

# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')

# Import KoalaRemoteClient
clr.AddReference("LynceeTec.KoalaRemote.Client")
from LynceeTec.KoalaRemote.Client import KoalaRemoteClient

# Get input form console, if input is empty us default value
def get_input(question,default) :
    answer = input(question+'['+default+'] :')
    return answer or default

# Define KoalaRemoteClient host
host=KoalaRemoteClient()

#Ask IP address
IP = 'localhost'

# Log on Koala - default to admin/admin combo
username = 'admin'
[ret,username] = host.Connect(IP,username,True);
# password = get_input('Enter password for '+username+' account', username)
host.Login('admin')

####################  Splitting of FastSave  ##################################
# Split the initial hologram if necessary (only needed for 'Fast' saved files)
# Will split ALL frames by default. If this splitting has already been done,
# then skip to the reconstruction methods

if HoloMethod == 'Fast':
    
    # Establish directories
    inputsequence = str(inputdirectory.absolute())
    outputsequences = str(SplittingDirectory.absolute())
    name_input_sequence=os.path.basename(os.path.normpath(inputsequence))
    FastFrameNums = [(0,NfastFrames)]

    # Check if splitting has already occured
    splitdir = Path(pathlib.PureWindowsPath(str(SplittingDirectory.absolute())).joinpath(name_input_sequence+'_'+str(FastFrameNums[0][0]).zfill(5)+'_TO_'+str(FastFrameNums[0][1]).zfill(5),'Holograms'))
    contents = splitdir.glob('*.tif')
    cnum = 0
    for i in contents:
        cnum = cnum + 1        
    print(str(cnum))
    
    if cnum != NfastFrames:    
        # Do splitting
        print('Splitting')
        HoloDirectory = Path(pathlib.PureWindowsPath(inputdirectory).joinpath('SplitHolograms'))
        HoloDirectory.mkdir(parents=True,exist_ok=True)
        holosize = 1024        
        # default_dir= r'C:/Users/Admin/Downloads/'
        #FrameNums=[(0,10),(30,50),(398,399)] #[(firstholo1,lastholo1),(firstholo2,lastholo2),...]
        
        # Call FastSplitter.py to do the splitting of the sequence
        for k in range(np.shape(FastFrameNums)[0]):
            outputsequence=outputsequences+'\\'+name_input_sequence+'_'+str(FastFrameNums[k][0]).zfill(5)+'_TO_'+str(FastFrameNums[k][1]).zfill(5)
            cropsequence = FastSplitter.Sequence(inputsequence,outputsequence,FastFrameNums[k][0],FastFrameNums[k][1],holosize)
            cropsequence.CreateOutputDirectory()
            cropsequence.DefineSeqFormat()
            cropsequence.saveCropSequence()
            cropsequence.WriteTimestamps()
            
    # Set final hologram directory for later
    HoloDirectory = Path(pathlib.PureWindowsPath(str(SplittingDirectory.absolute())).joinpath(name_input_sequence+'_'+str(FastFrameNums[0][0]).zfill(5)+'_TO_'+str(FastFrameNums[0][1]).zfill(5),'Holograms'))
    
if HoloMethod == 'Slow':
    HoloDirectory = Path(pathlib.PureWindowsPath(inputdirectory).joinpath('Holograms'))

# QC check - if the files haven't been assigned then no reconstruction!
if HoloMethod != 'Slow' and HoloMethod != 'Fast':
    print('Incorrect saving method - check and try again' + '\n')
    sys.exit()

####################  Main  ###################################################

globfiles = HoloDirectory.glob('*.tif')
hololist = []
for i in globfiles:
    hololist.append(i)

# Open config
host.OpenConfig(ConfigNumber);

# Open main display windows
host.OpenPhaseWin();
host.OpenIntensityWin();
host.OpenHoloWin();

# List of reconstruction heights used
reconDists = []

# Go through each holo frame, do reconstructions at planes, save images
for nf in range(FrameNums[0][0],FrameNums[0][1]):
    
    # Reset currentDistance to first reconstruction plane
    currDist = recDistLeft
    nplane = 0
        
    # Make output directories
    # Hologram
    outpath_H = Path(pathlib.PureWindowsPath(outputdirectory).joinpath("Hologram","Frame_" + str(nf).zfill(5)))
    outpath_H.mkdir(parents=True,exist_ok=True)
    # Phase
    outpath_P = Path(pathlib.PureWindowsPath(outputdirectory).joinpath("Phase","Frame_" + str(nf).zfill(5)))
    outpath_P.mkdir(parents=True,exist_ok=True)
    # Intensity
    outpath_I = Path(pathlib.PureWindowsPath(outputdirectory).joinpath("Intensity","Frame_" + str(nf).zfill(5)))
    outpath_I.mkdir(parents=True,exist_ok=True)
    
    # Load hologram
    holopath = Path(pathlib.PureWindowsPath(HoloDirectory).joinpath(hololist[nf]))
    print(str(holopath.absolute()))
    host.LoadHolo(str(holopath.absolute()),1)
    
    while currDist < recDistRight:
        
        # Move to the reconstruction distance
        host.SetRecDistCM(currDist); # Set distance
        host.OnDistanceChange(); # Move to that distance, do reconstruction
        
        save_H = Path(pathlib.PureWindowsPath(outpath_H).joinpath("Holo_Plane_" + str(nplane).zfill(5)))
        save_P = Path(pathlib.PureWindowsPath(outpath_P).joinpath("Phase_Plane_" + str(nplane).zfill(5)))
        save_I = Path(pathlib.PureWindowsPath(outpath_I).joinpath("Intensity_Plane_" + str(nplane).zfill(5)))
        
        # Unwrap phase
        host.SetUnwrap2DState(True);
        # Tilt correction/fitting
        # host.ComputePhaseCorrection(fitMethod,degreeOfFit)
        
        # Saving bin files 
        if SaveMethod == 0 or SaveMethod == 2:
            # Hologram
            print(str(save_H.absolute()) + ".bin")
            host.SaveImageFloatToFile(1,str(save_H.absolute()) + ".bin",True)
            # host.SaveImageFloatToFile(1,os.path.join(outpath_H, "Holo_Plane_" + str(nplane).zfill(5) + ".bin"),True);
            # Phase
            host.SaveImageFloatToFile(4,str(save_P.absolute()) + ".bin",True)
            # host.SaveImageFloatToFile(2,os.path.join(outpath_P, "Phase_Plane_" + str(nplane).zfill(5) + ".bin"),True);
            #Intensity
            host.SaveImageFloatToFile(2,str(save_I.absolute()) + ".bin",True)
            # host.SaveImageFloatToFile(4,os.path.join(outpath_I + "Intensity_Plane_" + str(nplane).zfill(5) + ".bin"),True);
                   
        # Saving tif files                   
        if SaveMethod == 1 or SaveMethod == 2:
            # Hologram
            host.SaveImageToFile(1,str(save_H.absolute()) + ".tif")
            # Phase
            host.SaveImageToFile(4,str(save_P.absolute()) + ".tif")
            # Intensity
            host.SaveImageToFile(2,str(save_I.absolute()) + ".tif")
            
        # Update distance/plane count
        if nf == 0:
            reconDists.append(currDist)
        currDist = currDist + zstep
        nplane = nplane + 1
            

# Logout of production mode
host.Logout();

####################  Logging  ################################################

# Files of the reconstruction heights used
reconDists.append(zstep)
textpath = Path(pathlib.PureWindowsPath(outputdirectory).joinpath("PlaneHeights.txt"))
with open(textpath.absolute(),'w') as f:
    for r in reconDists:
        f.write(str(r) + '\n')
    f.close()
        
# Metadata files
textpath = Path(pathlib.PureWindowsPath(outputdirectory).joinpath("Metadata.txt"))
with open(textpath.absolute(),'w') as f:    
    f.write('Experiment date: ' + ExpDate + '\n')
    f.write('Experiment Name: ' + ExpName + '\n')
    f.write('HoloMethod: ' + HoloMethod + '\n')
    f.write('Configuration Number: ' + str(ConfigNumber) + '\n')
    f.write('HoloDirectory: ' + str(HoloDirectory.absolute()) + '\n')
    f.write('SaveMethod: ' + str(SaveMethod) + '\n')
    f.write('Reconstruction range: ' + str(recDistLeft) + ' to ' + str(reconDists[-1]) + '\n')
    f.write('ztep: ' + str(zstep) + '\n')
    # f.write('DegreeOfFit: ' + str(degreeOfFit) + '\n')
    f.close()

# Copy timestamps file to same directory
textinputfile = Path(pathlib.PureWindowsPath(inputdirectory).joinpath('timestamps.txt'))
textoutputfile = Path(pathlib.PureWindowsPath(outputdirectory).joinpath('timestamps.txt'))
shutil.copyfile(str(textinputfile.absolute()),str(textoutputfile.absolute()))