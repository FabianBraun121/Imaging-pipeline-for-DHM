# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:28:24 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from spatial_phase_averaging import SpatialPhaseAveraging
loc_dir = r"C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\Sample2x2x36_forFabian\2023-02-28 10-06-34\00001"
from utils import connect_to_remote_koala
import time

#%%
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)

#%%
start = time.time()
loc_image = SpatialPhaseAveraging(loc_dir, 0, host)
end = time.time()

print(end-start)