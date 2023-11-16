# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:10:23 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from spatial_averaging.utilities import Koala
import numpy as np
import time
import matplotlib.pyplot as plt

image_fname1 = base_dir = os.getcwd() + os.sep + r'\..\data\test_data\00001\00001_00001\Holograms\00000_holo.tif'
image_fname2 = base_dir = os.getcwd() + os.sep + r'\..\data\test_data\00002\00001_00001\Holograms\00000_holo.tif'
image_fnames = [image_fname1, image_fname2]

Koala.connect(279)
host = Koala._host
host.LoadHolo(image_fnames[0],1)
host.SetUnwrap2DState(True)

evaluation_time = []
steps = 250
ev_per_step = 100
for i in range(steps):
    start = time.time()
    for j in range(ev_per_step):
        image_fname = image_fnames[j%2]
        host.LoadHolo(image_fname,1)
        host.SetUnwrap2DState(True)
        dist = np.random.uniform(-1, 1)
        Koala.set_reconstruction_distance(dist)
        _ = Koala.get_phase_image()
    evaluation_time.append(time.time()-start)
 #%%
plt.figure()
plt.plot(np.arange(steps)*ev_per_step, np.array(evaluation_time)/ev_per_step*1000)
plt.xlabel("n'th image evaluation")
plt.ylabel('time per image evaluation [ms]')
plt.show()
