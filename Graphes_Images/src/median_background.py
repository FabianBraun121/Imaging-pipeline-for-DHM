# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:55:09 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
from  pyKoalaRemote import client
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import minimize, Bounds
from scipy import ndimage
from skimage.registration import phase_cross_correlation


class Hologram:
    def __init__(self, fname):
        self.fname = fname
        self.reconstrution_distance_left = -3
        self.reconstrution_distance_right = -1
        self.focus = None # Focus distance
        self.focus_method = None # evaluation function
        self.focus_score = None # score of evaluatino function at the Focus point (minimum)
        self.optimizing_method = None # function descent method
        self.plane_basis_vectors = None # normally polynomial basis vectors 
        self.plane_fit_order = None # also named degree
        self.X_plane = None # since the evaluation points, aka imagepoints stay constant X for linear regression is always the same
        self.X_plane_pseudoinverse = None # together with X_plane here to calculate linear regression, less computational cost since constant for all images
        self.cache_path = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file'
        self.koala_host = None # client with which interaction with koala takes place
        self.cplx_image = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self, koala_host, focus_method='std_amp', optimizing_method= 'Powell', tolerance=None,
                        x0=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):

        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetUnwrap2DState(True)
        self.X_plane = self._X_plane()
        self.X_plane_pseudoinverse = self._X_plane_pseudoinverse()
        if x0 == None:
            x0 = (self.reconstrution_distance_left+self.reconstrution_distance_right)/2
        bounds = Bounds(lb=self.reconstrution_distance_left, ub=self.reconstrution_distance_right)
        res = minimize(self._evaluate_reconstruction_distance, [x0], method=self.optimizing_method, bounds=bounds)
        self.focus = res.x[0]
        self.focus_score = res.fun
        self.cplx_image = self._cplx_image(use_amp=use_amp)
            
    def _cplx_image(self, use_amp=True):
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetRecDistCM(self.focus)
        self.koala_host.OnDistanceChange()
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        ph, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        ph = self._subtract_plane(ph)
        if use_amp:
            self.koala_host.SaveImageFloatToFile(2,self.cache_path+'_amp.bin',True)
            amp, __header = binkoala.read_mat_bin(self.cache_path+'_amp.bin')
            cplx = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx = np.exp(complex(0.,1.)*ph)
        return cplx
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance):
        self.koala_host.SetRecDistCM(reconstruction_distance[0])
        self.koala_host.OnDistanceChange()
        if self.focus_method == 'std_amp':
            amp = self.koala_host.GetIntensity32fImage()
            return np.std(amp)
        elif self.focus_method == 'sobel_squared_std':
            ph = self.koala_host.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -self._evaluate_sobel_squared_std(ph)
        elif self.focus_method == 'combined':
            amp = self.koala_host.GetIntensity32fImage()
            ph = self.koala_host.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -np.std(ph)/np.std(amp)
        else:
            print("Method ", self.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_std_sobel_squared(self, gray_image):
        # Calculate gradient magnitude using Sobel filter
        grad_x = scipy.ndimage.sobel(gray_image, axis=0)
        grad_y = scipy.ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 2 + grad_y ** 2)
        
    def get_cplx_image(self):
        return self.cplx_image.copy()
    
    def header(self):
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        __ph, header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        return header
    
    def _subtract_plane(self, field):
        theta = np.dot(self.X_plane_pseudoinverse, field.reshape(-1))
        plane = np.dot(self.X_plane, theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _X_plane(self):
        if self.plane_basis_vectors == "Polynomial":
            ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
            ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
            X1, X2 = np.mgrid[:800, :800]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            return PolynomialFeatures(degree=self.plane_fit_order, include_bias=True).fit_transform(X)
        else:
            print(self.plane_basis_vectors,  ' is not implemented')
    
    def _X_plane_pseudoinverse(self):
        return np.dot( np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())

class SpatialPhaseAveraging:
    def __init__(self, loc_dir, timestep, koala_host, median_background=False, subtract_bacteria=True, focus_method='std_amp', optimizing_method= 'Powell',
                 tolerance=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):
        self.median_background = median_background
        self.subtract_bacteria = subtract_bacteria
        self.loc_dir = loc_dir
        self.timestep = timestep
        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.tolerance = tolerance
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.use_amp = use_amp
        self.x0_guess = None
        self.pos_list = [ f.name for f in os.scandir(loc_dir) if f.is_dir()]
        self.num_pos = len(self.pos_list)
        self.focus_score_list = []
        self.cplx_images_list = []
        self.holo_list = self._generate_holo_list()
        self.holo_list_in_use = self.holo_list
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    
    def _background(self):
        if self.median_background:
            background = np.zeros((len(self.holo_list_in_use),800,800), dtype=np.complex128)
            for i in range(0, len(self.holo_list_in_use)):
                background[i] = self.holo_list_in_use[i].get_cplx_image()
                
            return np.median(np.abs(background), axis=0)*np.exp(1j*np.median(np.angle(background), axis=0))
        else:
            background = self.holo_list_in_use[0].get_cplx_image()
            if self.subtract_bacteria:
                background = self._subtract_bacterias(background)
            for i in range(1, len(self.holo_list_in_use)):
                cplx_image = self.holo_list_in_use[i].get_cplx_image()
                cplx_image = self._subtract_phase_offset(cplx_image, background)
                if self.subtract_bacteria:
                    cplx_image = self._subtract_bacterias(cplx_image)
                background += cplx_image
            return background/len(self.holo_list_in_use)
    
    def _cplx_avg(self):
        cplx_avg = self.holo_list_in_use[0].get_cplx_image()
        cplx_avg /= self.background
        self.cplx_images_list.append(cplx_avg)
        for i in range(1, len(self.holo_list_in_use)):
            cplx_image = self.holo_list_in_use[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image = self._shift_image(cplx_avg, cplx_image)
            cplx_image = self._subtract_phase_offset(cplx_image, cplx_avg)
            cplx_avg += cplx_image
            self.cplx_images_list.append(cplx_image)
        return cplx_avg/len(self.holo_list_in_use)
    
    def _generate_holo_list(self):
        holo_list = []
        for pos in self.pos_list:
            fname = self.loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif"
            holo = Hologram(fname)
            holo.calculate_focus(self.koala_host, focus_method=self.focus_method, optimizing_method=self.optimizing_method, tolerance=self.tolerance,
                                 x0=self.x0_guess, plane_basis_vectors=self.plane_basis_vectors, plane_fit_order=self.plane_fit_order)
            # first guess is the focus point of the last image
            self.x0_guess = holo.focus
            self.focus_score_list.append(holo.focus_score)
            holo_list.append(holo)
        return holo_list
    
    def get_amp_avg(self):
        return np.absolute(self.cplx_avg)
    
    def get_cplx_avg(self):
        return self.cplx_avg.copy()
    
    def get_mass_avg(self):
        ph = np.angle(self.cplx_avg)
        cut_off = 0.15
        return np.sum(ph[cut_off<ph])
    
    def get_phase_avg(self):
        return np.angle(self.cplx_avg)
    
    def restrict_holo_use(self, holo_used):
        self.holo_list_in_use = [self.holo_list[i] for i in holo_used]
        self.num_pos = len(holo_used)
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    def _shift_image(self, reference_image, moving_image):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shiftVector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shiftVector, mode='wrap')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shiftVector, mode='wrap')
        return real+complex(0.,1.)*imaginary
    
    def _subtract_bacterias(self, cplx_image):
        # subtracts pixel  that are far away from the mean and replaces them with the mean of the image
        # cut off value is determined by hand and has to be reestimated for different use cases
        cut_off = 0.15
        ph = np.angle(cplx_image)
        ph[cut_off<ph] = np.mean(ph[ph<cut_off])
        return np.absolute(cplx_image)*np.exp(1j*ph)
    
    def _subtract_phase_offset(self, new, avg):
        z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new
    
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

#%%
base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
data_path = base_path + r'\data\median_background\2023-02-28 10-06-34\00001'
save_path = base_path + r'\Graphes_Images\median_background'
#%%
########################## start Koala and define functions ##########################
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)
#%%
spa_median = SpatialPhaseAveraging(data_path, 0, host, median_background=True)
background_median = spa_median.background
avg_median = spa_median.get_cplx_avg()

spa_mean_with_bacterias = SpatialPhaseAveraging(data_path, 0, host, median_background=False, subtract_bacteria=False)
background_mean_with_bacterias = spa_mean_with_bacterias.background
avg_mean_with_bacterias = spa_mean_with_bacterias.get_cplx_avg()

spa_mean_without_bacterias = SpatialPhaseAveraging(data_path, 0, host, median_background=False, subtract_bacteria=True)
background_mean_without_bacterias = spa_mean_without_bacterias.background
avg_mean_without_bacterias = spa_mean_without_bacterias.get_cplx_avg()

#%%
np.save(save_path + os.sep + 'background_mean_with_bacterias', background_mean_with_bacterias)
np.save(save_path + os.sep + 'avg_mean_with_bacterias', avg_mean_with_bacterias)
np.save(save_path + os.sep + 'background_mean_without_bacterias', background_mean_without_bacterias)
np.save(save_path + os.sep + 'avg_mean_without_bacterias', avg_mean_without_bacterias)
np.save(save_path + os.sep + 'background_median', background_median)
np.save(save_path + os.sep + 'avg_median', avg_median)

#%%

background_mean_with_bacterias = np.load(save_path + os.sep + 'background_mean_with_bacterias.npy')
avg_mean_with_bacterias = np.load(save_path + os.sep + 'avg_mean_with_bacterias.npy')
background_mean_without_bacterias = np.load(save_path + os.sep + 'background_mean_without_bacterias.npy')
avg_mean_without_bacterias = np.load(save_path + os.sep + 'avg_mean_without_bacterias.npy')
background_median = np.load(save_path + os.sep + 'background_median.npy')
avg_median = np.load(save_path + os.sep + 'avg_median.npy')

#%%

plt.figure('background mean_with_bacterias - median')
plt.imshow(np.angle(background_mean_with_bacterias)-np.angle(background_median))
plt.figure('background mean_without_bacterias - median')
plt.imshow(np.angle(background_mean_without_bacterias)-np.angle(background_median))

plt.figure('avg_median')
plt.imshow(np.angle(avg_median))
plt.figure('avg_mean_without_bacterias')
plt.imshow(np.angle(avg_mean_without_bacterias))

