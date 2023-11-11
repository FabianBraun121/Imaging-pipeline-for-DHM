# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:27 2023

@author: SWW-Bc20
"""
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds
import scipy.ndimage

from . import binkoala
from . import utilities as utils
from .utilities import cfg

class Hologram:
    def __init__(self, fname, roi):
        self.fname = fname
        self.roi = roi
        self.focus = None # Focus distance
        self.focus_score = None # score of evaluatino function at the Focus point (minimum)
        self.X_plane = None # since the evaluation points, aka imagepoints stay constant X for linear regression is always the same
        self.X_plane_pseudoinverse = None # together with X_plane here to calculate linear regression, less computational cost since constant for all images
        self.cplx_image = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self, x0=None):
        cfg.KOALA_HOST.LoadHolo(self.fname,1)
        cfg.KOALA_HOST.SetUnwrap2DState(True)
        self.X_plane = self._X_plane()
        self.X_plane_pseudoinverse = self._X_plane_pseudoinverse()
        if x0 == None:
            x0 = (cfg.reconstrution_distance_low+cfg.reconstrution_distance_high)/2
        bounds = Bounds(lb=cfg.reconstrution_distance_low, ub=cfg.reconstrution_distance_high)
        res = minimize(self._evaluate_reconstruction_distance, [x0], method=cfg.optimizing_method, bounds=bounds)
        self.focus = res.x[0]
        self.focus_score = res.fun
        self.cplx_image = self._cplx_image()
            
    def _cplx_image(self):
        cfg.KOALA_HOST.LoadHolo(self.fname,1)
        cfg.KOALA_HOST.SetRecDistCM(self.focus)
        cfg.KOALA_HOST.OnDistanceChange()
        ph = cfg.KOALA_HOST.GetPhase32fImage()
        ph = self._subtract_plane(ph)
        if cfg.use_amp:
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            cplx_image = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx_image = np.exp(complex(0.,1.)*ph)
        return cplx_image
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance):
        cfg.KOALA_HOST.SetRecDistCM(reconstruction_distance[0])
        cfg.KOALA_HOST.OnDistanceChange()
        ph = cfg.KOALA_HOST.GetPhase32fImage()
        ph = self._subtract_plane(ph)
        if cfg.focus_method == 'Max_std_of_phase_squard':
            # saves the phase of the hologram
            ph *= ph
            #returns the negativ since most optimizing function look for the Minimum
            return -np.std(ph)
        elif cfg.focus_method == 'sobel_squared_std':
            return -self._evaluate_sobel_squared_std(ph)
        else:
            print("Method ", cfg.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_sobel_squared_std(self, gray_image):
        # Calculate gradient magnitude using Sobel filter
        grad_x = scipy.ndimage.sobel(gray_image, axis=0)
        grad_y = scipy.ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 2 + grad_y ** 2)
    
    def get_cplx_image(self):
        return self.cplx_image.copy()
    
    def _subtract_plane(self, field):
        theta = np.dot(self.X_plane_pseudoinverse, field.reshape(-1))
        plane = np.dot(self.X_plane, theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _X_plane(self):
        if cfg.plane_basis_vectors == "Polynomial":
            ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
            ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
            X1, X2 = np.mgrid[:cfg.image_size[0], :cfg.image_size[1]]
            X1, X2 = X1[self.roi==True], X2[self.roi==True]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            return PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X)
        else:
            print(cfg.plane_basis_vectors,  ' is not implemented')
    
    def _X_plane_pseudoinverse(self):
        return np.dot( np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())