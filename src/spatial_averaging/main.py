# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:12:04 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from utils import connect_to_remote_koala, Open_Directory, get_result_unwrap
from spatial_phase_averaging import SpatialPhaseAveraging
import binkoala
import numpy as np
import time


start = time.time()
ConfigNumber = 218
host = connect_to_remote_koala(ConfigNumber)
default_dir = r'Q:\SomethingFun' 
base_dir = Open_Directory(default_dir, "Open a scanning directory")
save_base_folder = base_dir + " phase averages"
if not os.path.exists(save_base_folder):
    os.makedirs(save_base_folder)
all_loc = [ f.name for f in os.scandir(base_dir) if f.is_dir()]
timestamps = len(os.listdir(base_dir+os.sep+all_loc[0]+os.sep+"00001_00001\Holograms"))
#%%
for loc in all_loc[1:]:
    for i in range(timestamps)[:171]:
        loc_dir = base_dir+os.sep+loc
        spa = SpatialPhaseAveraging(loc_dir, i, host)
        averaged_image = spa.get_cplx_avg()
        ph = get_result_unwrap(np.angle(averaged_image))
        
        save_loc_folder = save_base_folder +os.sep + loc
        if not os.path.exists(save_loc_folder):
            os.makedirs(save_loc_folder)
        fname = save_loc_folder +"\\timestep_"+str(i).zfill(5)+".bin"
        header = spa.holo_list[0].header()
        width = (int)(header["width"])
        height = (int)(header["height"])
        px_size = (float)(header["px_size"])
        hconv = (float)(header["hconv"])
        unit_code = (int)(header["unit_code"])
        binkoala.write_mat_bin(fname, ph, width, height, px_size, hconv, unit_code) 
        print(fname)

end = time.time()
print((end-start)//60, ' min')



