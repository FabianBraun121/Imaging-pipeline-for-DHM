# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:15:54 2023

@author: SWW-Bc20
"""

import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from utils import connect_to_remote_koala, Open_Directory, get_result_unwrap
from hologram import Hologram
import binkoala
import numpy as np
import time

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\check_focusing'
if not os.path.exists(save_path):
    os.makedirs(save_path)

start = time.time()
ConfigNumber = 219
host = connect_to_remote_koala(ConfigNumber)
default_dir = r'Q:\SomethingFun' 
base_dir = Open_Directory(default_dir, "Open a scanning directory")
all_loc = [ f.name for f in os.scandir(base_dir) if f.is_dir()]
timesteps = len(os.listdir(base_dir+os.sep+all_loc[0]+os.sep+"00001_00001\Holograms"))
num_images = 100
images = np.zeros((num_images,800,800))
focus_distances = np.zeros((num_images))
fnames = []

for i in range(100):
    l = np.random.randint(len(all_loc))
    loc = all_loc[l]
    loc_dir = base_dir+os.sep+loc
    positions = os.listdir(loc_dir)
    pos = positions[np.random.randint(len(positions))]
    timestep = np.random.randint(timesteps)
    fname= loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(timestep).zfill(5) + "_holo.tif"
    fnames.append(fname)
    holo = Hologram(fname)
    holo.calculate_focus(host)
    focus_distances[i] = holo.focus
    images[i] = np.angle(holo.get_cplx_image())

fnames = np.array(fnames)
#%%

def get_mass(images):
    cut_off = 0.25
    mass = np.zeros((images.shape[0]))
    for i in range(images.shape[0]):
        mass[i] = np.sum(images[i][cut_off<images[i]])
    return mass
mass = get_mass(images)
print('images with low mass: ', np.arange(num_images)[mass<100])
#%%
i = 24
host.LoadHolo(str(fnames[i]),1)
host.SetRecDistCM(focus_distances[i])
host.OnDistanceChange()