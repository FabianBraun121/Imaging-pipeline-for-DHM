# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:04:01 2023

@author: SWW-Bc20
"""

"""
summary of results:
Test showed that the Powell method was clearly the best. Many optimizer dont work
without having the hessian or being linear. Some optimizing methods can also not use the bounds.
Calculating one results takes in average 7 seconds,if the starting position is well defined it
drops to around 4 seconds. Still this is a significant time, so it should be further tested if it
is possible to decrease tolerance to achive faster time without lowering significant quality

The tolerance of the optimizer is significantly smaller as the tolerance of the control. So +- 0.01cm
is deemed the rigth answer.
"""
import clr
import sys
import binkoala
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\optimizer_methods'
optimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA','SLSQP', 'trust-constr']

#%%
########################## start Koala and define functions ##########################
# Add Koala remote librairies to Path
sys.path.append(r'C:\Program Files\LynceeTec\Koala\Remote\Remote Libraries\x64')

# Import KoalaRemoteClient
clr.AddReference("LynceeTec.KoalaRemote.Client")
from LynceeTec.KoalaRemote.Client import KoalaRemoteClient

ConfigNumber = 221
# Define KoalaRemoteClient host
host=KoalaRemoteClient()

#Ask IP address
IP = 'localhost'

# Log on Koala - default to admin/admin combo
username = 'admin'
[ret,username] = host.Connect(IP,username,True);
host.Login('admin')
# Open config
host.OpenConfig(ConfigNumber);
host.OpenPhaseWin()
host.OpenIntensityWin()
host.OpenHoloWin()
fname = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\Sample2x2x36_forFabian\2023-02-28 10-06-34\00001\00001_00001\Holograms\00000_holo.tif'
host.LoadHolo(fname,1)

def subtract_plane(field, plane_degree):
    X1, X2 = np.mgrid[:field.shape[0], :field.shape[1]]
    X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
    X = PolynomialFeatures(degree=plane_degree, include_bias=False).fit_transform(X)
    y = field.reshape(-1)
    reg = LinearRegression().fit(X, y)
    plane = reg.predict(X).reshape(field.shape[0],field.shape[1])
    return field - plane

def evaluate_reconstruction_distance_minus_squared_std(x):
    host.SetRecDistCM(x[0])
    host.OnDistanceChange()
    host.SaveImageFloatToFile(4,r'C:\Master_Thesis_Fabian_Braun\Data\test.bin',True)
    image_values, header = binkoala.read_mat_bin(r'C:\Master_Thesis_Fabian_Braun\Data\test.bin')
    image_values = subtract_plane(image_values, 1)
    image_values *= image_values
    return -np.std(image_values)

#%%
########################## calculate whole function space ##########################
start = time.time()
x = np.zeros(400)
y = np.zeros(400)
for i in range(400):
    xi = -3+i*0.01
    x[i] = xi
    y[i] = -evaluate_reconstruction_distance_minus_squared_std([xi])
np.save(save_path+'/x', x)
np.save(save_path+'/y', y)
end = time.time()
print(end-start)
#%%
########################## run test on different optimizing methods, save results ##########################
import warnings
warnings.filterwarnings("ignore")
starts = np.arange(-3,1,0.1)
results = np.ndarray((len(optimizers), len(starts)))
duration = np.ndarray((len(optimizers), len(starts)))
bounds = Bounds(lb=-3, ub=1, keep_feasible=False)
for i, optimizer in enumerate(optimizers):
    for j in range(len(starts)):
        start = time.time()
        res = minimize(evaluate_reconstruction_distance_minus_squared_std, [starts[j]], method=optimizer, bounds=bounds)
        results[i,j] = res.x[0]
        end = time.time()
        duration[i,j] = end-start
    print(optimizer, ' done')
np.save(save_path+'/results', results)
np.save(save_path+'/duration', duration)
#%%
########################## load results of test ##########################
x = np.load(save_path+'/x.npy')
y = np.load(save_path+'/y.npy')
results = np.load(save_path+'/results.npy')
duration = np.load(save_path+'/duration.npy')

#%%
########################## plot ##########################
score = (np.abs(results-x[np.argmax(y)])<0.01).sum(axis=1)
plt.figure()
for i in range(results.shape[0]):
    plt.plot(np.arange(-3,1,0.1), results[i], label=f"{optimizers[i]}, score: {score[i]}, avg time: {duration.mean(axis=1)[i]:.2f}")
plt.legend()
plt.title("optimizing methods with 40 different starting positions")
plt.show()

