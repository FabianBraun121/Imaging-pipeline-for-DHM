# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:27 2023

@author: SWW-Bc20
"""
import os
import binkoala
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds
import scipy.ndimage

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
        self.cache_path = os.getcwd() + os.sep + '..\\..\\data\\__file'
        self.koala_host = None # client with which interaction with koala takes place
        self.cplx_image = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self, koala_host, focus_method='sharpness_squared_std', optimizing_method= 'Powell', tolerance=None,
                        x0=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):
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
            cplx_image = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx_image = np.exp(complex(0.,1.)*ph)
        return cplx_image
    
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
        elif self.focus_method == 'sharpness_squared_std':
            return -self._evaluate_sobel_squared_std(image_values)
        else:
            print("Method ", self.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_sobel_squared_std(self, gray_image):
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
        """

        Returns
        -------
        numpy array
            DESCRIPTION. complex image

        Method Description
        -------
        returns the complex image at the focus point. Function should be used instead of obj.cplx_image
        since this copies the image. Otherwise image changed outside the function would influence cplx_image
        
        """
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