# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:05:44 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from pyKoalaRemote import client
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging\tests')

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\measure_koala_duration_over_time'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
ConfigNumber = 219
# Define KoalaRemoteClient host
host = client.pyKoalaRemoteClient()
host.Connect('localhost')
host.Login('admin')
# Open config
host.OpenConfig(ConfigNumber);
host.OpenPhaseWin()
host.OpenIntensityWin()
host.OpenHoloWin()
#%%

num_images = 1000
num_evaluations_per_image = 100
duration = np.zeros(num_images)
holograms = [ fname for fname in os.listdir(save_path) if fname.endswith('.tif')]

test_start = time.time()
for i in range(num_images):
    start = time.time()
    fname = save_path + os.sep + holograms[random.randint(0,len(holograms)-1)]
    host.LoadHolo(fname,1)
    for j in range(num_evaluations_per_image):
        host.SetRecDistCM(random.random())
        host.OnDistanceChange()
        int_values = host.GetIntensity32fImage()
        ph_values = host.GetPhase32fImage()
    duration[i] = time.time()-start
    if i%10 == 0:
        print(f'image number {i} is done!')
print(f'experiment took {np.round((time.time()-test_start)/60,1)} minutes!')

np.save(save_path + os.sep + 'duration', duration)

#%%
duration = np.load(save_path + os.sep + 'duration.npy')

#%%
plt.plot(np.arange(duration.shape[0])*100+1, duration*10)
plt.xticks(np.arange(6)*20000, [f'{2*i}*$10^4$' for i in range(6)])
plt.xlabel('$n^{th}$ image evaluation')
plt.ylabel('duration [ms]')

