# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:36 2023

@author: SWW-Bc20
"""

import os
import sys
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
from utils import connect_to_remote_koala
import binkoala
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from utils import connect_to_remote_koala, Open_Directory, get_result_unwrap
import cv2
import scipy.ndimage

save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\spatial_averaging\holo_in_use'
if not os.path.exists(save_path):
    os.makedirs(save_path)


#%%

def evaluate_std_sobel_squared(gray_image):
    # Calculate gradient magnitude using Sobel filter
    grad_x = scipy.ndimage.sobel(gray_image, axis=0)
    grad_y = scipy.ndimage.sobel(gray_image, axis=1)
    grad_mag_squared = grad_x ** 2 + grad_y ** 2
    
    # Calculate sharpness score as the std gradient magnitude
    sharpness = np.std(grad_mag_squared)
    
    return sharpness


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
    
    def calculate_focus(self, koala_host, focus_method='std_sobel_squared', optimizing_method= 'Powell', tolerance=None,
                        x0=None, plane_basis_vectors='Polynomial', plane_fit_order=3, use_amp=True):
        """
        
        Parameters
        ----------
        koala_host : KoalaClient
            DESCRIPTION. koala client.
        focus_method : string, optional
            DESCRIPTION. The default is 'sharpness_squared_std'.
        optimizing_method : string, optional
            DESCRIPTION. The default is 'Powell'.
        tolerance : float or None, optional
            DESCRIPTION. The default is None.
        x0 : float or None, optional
            DESCRIPTION. The default is None.
        plane_basis_vectors : string, optional
            DESCRIPTION. The default is 'Polynomial'.
        plane_fit_order : int<=1, optional
            DESCRIPTION. The default is 3.
        use_amp : boolean, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.
        
        Method Description
        -------
        Calculates the minimum negative sharpness to find ideal focus point.
        Calculates complex image at the focus point.
        
        """
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
        """

        Parameters
        ----------
        use_amp : Boolean
            DESCRIPTION. The default is True.

        Returns
        -------
        cplx : numpy array
            DESCRIPTION.
        
        Method Description
        -------
        Calculates complex image at the focus point by
        1. Read out phase from Koala
        2. Subtract plane calculated with linear regression
        3. Read out amplitude and combine with pahse

        """
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
        """

        Parameters
        ----------
        reconstruction_distance : float
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.
        
        Method Description
        -------
        evaluates the image quality of the given focus distance.
        1. Read out phase from Koala
        2. Subtract plane calculated with linear regression
        3. Evaluation of image sharpness with self.focus_method

        """
        self.koala_host.SetRecDistCM(reconstruction_distance[0])
        self.koala_host.OnDistanceChange()
        # saves the phase of the hologram
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        image_values, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        image_values = self._subtract_plane(image_values)
        if self.focus_method == 'Max_std_of_phase_squard':
            # saves the phase of the hologram
            image_values *= image_values
            #returns the negativ since most optimizing function look for the Minimum
            return -np.std(image_values)
        elif self.focus_method == 'std_sobel_squared':
            return -self._evaluate_std_sobel_squared(image_values)
        else:
            print("Method ", self.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_std_sobel_squared(self, gray_image):
        """

        Parameters
        ----------
        gray_image : numpy array
            DESCRIPTION. image with only one "color" channel

        Returns
        -------
        float
            DESCRIPTION. "sharpness" score
        
        Method Description
        -------
        metric to calculate image sharpness. sobel kernel registers change in x or
        y direction. sqrt(grad_x^2 + grad^2) gradient magnitude. Square magnitude to increase
        strong changes (like a boundry of a bacteria). Images with high variations in squared 
        gradient magnitude have high standard deviations, aka clear bacterias

        """
        # Calculate gradient magnitude using Sobel filter
        grad_x = scipy.ndimage.sobel(gray_image, axis=0)
        grad_y = scipy.ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 2 + grad_y ** 2)
        
    def get_cplx_image(self):
        return self.cplx_image.copy()
    
    def header(self):
        """

        Returns
        -------
        header : header object
            DESCRIPTION. 
        
        Method Description
        -------
        Reads out the header of an image and returns it. These information are needed to save all
        derivative images.
        
        """
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        __ph, header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        return header
    
    def _subtract_plane(self, field):
        """

        Parameters
        ----------
        field : numpy array
            DESCRIPTION. Is most of the time the phase image

        Returns
        -------
        numpy array
            DESCRIPTION. Image subtracted by plane
        
        Method Description
        -------
        When images are taken in Koala they are not always exactly on flat surfaces any tilt or small
        roundness is corrected for with this method
        
        """
        theta = np.dot(self.X_plane_pseudoinverse, field.reshape(-1))
        plane = np.dot(self.X_plane, theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _X_plane(self):
        """

        Returns
        -------
        numpy array
            DESCRIPTION. Returns basis vectors of Image
            
        Method Description
        -------
        An ideal plane from the basis vectors can be calculated with linear regression.
        For that the two dimensional image grid needs to be flatted and extended to chosen degree.
        
        """
        if self.plane_basis_vectors == "Polynomial":
            ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
            ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
            X1, X2 = np.mgrid[:800, :800]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            return PolynomialFeatures(degree=self.plane_fit_order, include_bias=True).fit_transform(X)
        else:
            print(self.plane_basis_vectors,  ' is not implemented')
    
    def _X_plane_pseudoinverse(self):
        """

        Returns
        -------
        numpy array
            DESCRIPTION. Moore-Penrose-Inverse

        Method Description
        -------
        Calculating this takes a "long" time since an inverse is present. Since this
        is the same for all images, computational cost is saved by only calculating it once.
        
        """
        return np.dot( np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())


class SpatialPhaseAveraging:
    def __init__(self, loc_dir, timestep, koala_host, focus_method='std_sobel_squared', optimizing_method= 'Powell',
                 tolerance=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):
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
        self.holo_list = self._generate_holo_list()
        self.holo_list_in_use = self.holo_list
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    
    def _background(self):
        background = self.holo_list_in_use[0].get_cplx_image()
        background = self._subtract_bacterias(background)
        for i in range(1, len(self.holo_list_in_use)):
            cplx_image = self.holo_list_in_use[i].get_cplx_image()
            cplx_image = self._subtract_phase_offset(cplx_image, background)
            cplx_image = self._subtract_bacterias(cplx_image)
            background += cplx_image
        return background/len(self.holo_list_in_use)
    
    def _cplx_avg(self):
        cplx_avg = self.holo_list_in_use[0].get_cplx_image()
        cplx_avg /= self.background
        for i in range(1, len(self.holo_list_in_use)):
            cplx_image = self.holo_list_in_use[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image = self._shift_image(cplx_avg, cplx_image)
            cplx_image = self._subtract_phase_offset(cplx_image, cplx_avg)
            cplx_avg += cplx_image
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
    
#%%
ConfigNumber = 219
host = connect_to_remote_koala(ConfigNumber)
default_dir = r'Q:\SomethingFun' 
base_dir = Open_Directory(default_dir, "Open a scanning directory")
all_loc = [ f.name for f in os.scandir(base_dir) if f.is_dir()]
timestamps = len(os.listdir(base_dir+os.sep+all_loc[0]+os.sep+"00001_00001\Holograms"))
#%%
all_holos = list(np.arange(25))
every_secand = list(np.arange(0,25,2))
X_form = [0,4,6,8,12,16,18,20,24]
tree_square = [0,1,2,5,6,7,10,11,12]
four_square = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]
edges = [0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24]
three_square_outer_every_second = [0,2,4,6,7,8,10,11,12,13,14,16,17,18,20,22,24]
two_square_and_edges = [0,3,6,7,11,12,15,18]
edge_every_secand = [0,2,4,10,14,20,22,24]
num_images = 100
num_images_saved = 20
holos_in_use = [all_holos, every_secand, X_form, tree_square, four_square, edges, three_square_outer_every_second, two_square_and_edges, edge_every_secand]
holos_name_list = ['all_holos', 'every_secand', 'X_form', 'tree_square', 'four_square', 'edges', 'three_square_outer_every_second', 'two_square_and_edges', 'edge_every_secand']
std_sobel_squared = np.zeros((num_images,len(holos_in_use)))
mass = np.zeros((num_images,len(holos_in_use)))
std_diff_to_full_image = np.zeros((num_images,len(holos_in_use)))
focus_score_lists = []
exmaplary_images = np.ndarray((num_images_saved,len(holos_in_use), 800, 800))

#%%
start = time.time()
for i in range(num_images):
    l = np.random.randint(len(all_loc))
    loc = all_loc[l]
    timestamp = np.random.randint(timestamps)
    loc_dir = base_dir+os.sep+loc
    spa = SpatialPhaseAveraging(loc_dir, timestamp , host)
    focus_score_lists.append(spa.focus_score_list)
    phase_all = get_result_unwrap(np.angle(spa.get_cplx_avg()))
    for j, holo_in_use in enumerate(holos_in_use):
        spa.restrict_holo_use(holo_in_use)
        averaged_image = spa.get_cplx_avg()
        ph = get_result_unwrap(np.angle(averaged_image))
        if i<num_images_saved:
            exmaplary_images[i,j] = ph
        std_sobel_squared[i,j] = evaluate_std_sobel_squared(ph[100:700,100:700])
        mass[i,j] = spa.get_mass_avg()
        std_diff_to_full_image[i,j] = np.std(ph[100:700,100:700]-phase_all[100:700,100:700])
    print(i, " done")
end = time.time()
print(f'evaluation took {(end-start)//60} min, which is {np.round((end-start)/num_images,1)} secands per image')

focus_score = np.array(focus_score_lists)
holos_name = np.array(holos_name_list)
in_use = np.array([len(holos_in_use[i]) for i in range(len(holos_in_use))])
np.save(save_path + r'\std_sobel_squared', std_sobel_squared)
np.save(save_path + r'\mass', mass)
np.save(save_path + r'\std_diff_to_full_image', std_diff_to_full_image)
np.save(save_path + r'\focus_score', focus_score)
np.save(save_path + r'\in_use', in_use)
np.save(save_path + r'\holos_name', holos_name)
np.save(save_path + r'\exmaplary_images', exmaplary_images)
#%%
std_sobel_squared = np.load(save_path + r'\std_sobel_squared.npy')
mass = np.load(save_path + r'\mass.npy')
std_diff_to_full_image = np.load(save_path + r'\std_diff_to_full_image.npy')
std_sobel_squared = np.load(save_path + r'\std_sobel_squared.npy')
focus_score = np.load(save_path + r'\focus_score.npy')
in_use = np.load(save_path + r'\in_use.npy')
holos_name = np.load(save_path + r'\holos_name.npy')
exmaplary_images = np.load(save_path + r'\exmaplary_images.npy')
#%%
mass_ = mass[1000<mass[:,0]]
mass_change = (mass_[:,1:].T/mass_[:,0]).T
std_sobel_squared_ = std_sobel_squared[1000<mass[:,0]]
sharpness_change = (std_sobel_squared_[:,1:].T/std_sobel_squared_[:,0]).T
labels = []
for i in range(len(holos_name)-1):
    labels.append(f'{holos_name[i+1]}\n{in_use[i+1]}')

plt.figure('sharpness comparison')
plt.boxplot(sharpness_change, labels=labels)
plt.savefig(save_path+"/sharpness_comparison", dpi=300)
plt.show()

plt.figure('mass comparison')
plt.boxplot(mass_change, labels=labels)
plt.savefig(save_path+"/mass_comparison", dpi=300)
plt.show()

#%%
normalized_images = exmaplary_images[:,:,100:700,100:700].copy()
normalized_images[:,1:] = np.array([normalized_images[i,1:]-normalized_images[i,0] for i in range(normalized_images.shape[0])])
std = np.mean(std_diff_to_full_image, axis=(0))
#%%
labels = [f'{in_use[i]}: {holos_name[i]}, std: {np.round(std[i]*1000,3)}' for i in range(len(std))]
image = 9
horizontal_images = len(holos_in_use)//2
vertical_images = 2
fig , ax = plt.subplots(vertical_images,horizontal_images)
for i in range(vertical_images):
    for j in range(horizontal_images):
        k = i*horizontal_images+j+1
        ax[i,j].imshow(normalized_images[image, k])
        label = f'{in_use[k]}: {holos_name[k]}, std: {np.round(np.std(normalized_images[image,k]*1000),3)}*e-3'
        ax[i,j].set_title(label)
            

