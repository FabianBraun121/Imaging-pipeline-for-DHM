# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:24:04 2023

@author: SWW-Bc20
"""
import os
import sys
import numpy as np
import gc
import time
import numpy.typing as npt
import cv2
import tifffile
import json
import skimage.transform as trans
from datetime import datetime
from typing import List, Union, Tuple
from pathlib import Path
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds

sys.path.append("..")
from .utilities import (
    get_result_unwrap, get_masks_corners, gradient_squared, grid_search_2d,
    logout_login_koala,shut_down_restart_koala, start_koala, crop_image, zoom
)
import config as cfg

from . import binkoala

Image = npt.NDArray[np.float32]

CplxImage = npt.NDArray[np.complex64]

Mask = npt.NDArray[np.uint8]


class Position:
    """
    In each experiment different positions of the sample are observed. In this class informations of induvidual positions are shared here.
    Order: Experment -> Position -> Placement -> Timestep
    """
    def __init__(
            self,
            pos_dir: Union[str, Path],
            ):
        """
        Parameters
        ----------
        pos_dir : Union[str, Path]
            Path into the folder of induvidual position. Folders of the different placements are expected in this folder.

        Returns
        -------
        None.

        """
        
        self.pos_dir: Path = Path(pos_dir)
        "Path to position"
        self.pos_name: str = pos_dir.name
        "position name, e.g. 00001"
        self.image_roi_selected: bool = False
        "checks if roi is selected"
        self.pos_image_roi: Mask = np.ones(cfg.image_size, dtype=np.uint8)
        "surface plane calculation only on this part of the image"
        self.recon_rectangle_selected: bool = False
        "checks if a reconstructoin rectangle (used for focusing) is selected"
        self.pos_recon_corners: Tuple[Tuple[int]] = None  #((ymin, ymax), (xmin, xmax))
        "focusing only inside this rectangle"
        self.backgrounds = []
        "list for storage of the first 10 backgrounds"
        self.background = self._background()
        "if background exists its loaded here"
        
        
    def average_backgrounds(self):
        "averages self.backgrounds and saves this image"
        self.background = np.mean(np.abs(self.backgrounds), axis=0) * np.exp(1j* np.mean(np.angle(self.backgrounds), axis=0)).astype(np.complex64)
        tifffile.imwrite(str(self.pos_dir)+os.sep+'background.tif', self.background)
    
    def _background(self):
        "checks whether there is a background already calculated for this Position, if so it is loaded"
        if os.path.isfile(str(self.pos_dir)+os.sep+'background.tif'):
            return np.array(tifffile.imread(str(self.pos_dir)+os.sep+'background.tif'))
        else:
            return None
            
    def get_background(self):
        return self.background.copy()
    
    def get_pos_image_roi(self) -> Mask:
        return self.pos_image_roi.copy()
    
    def get_pos_image_roi_corners(self) -> Tuple[Tuple[int]]:
        "returns outermost corners of the roi mask"
        if self.image_roi_selected:
            return get_masks_corners(self.pos_image_roi)
        else:
            return [[0,cfg.image_size[0]],[0,cfg.image_size[1]]]
    
    def set_pos_image_roi(self, pos_image_roi: Mask):
        self.pos_image_roi = pos_image_roi.astype(np.uint8)
        self.image_roi_selected = True
    
    def get_pos_recon_corners(self) -> Tuple[Tuple[int]] :
        "returns reconstruction corners, if none are selected roi cornes are used."
        if self.pos_recon_corners is None:
            return self.get_pos_image_roi_corners()
        else:
            return self.pos_recon_corners
    
    def set_pos_recon_corners(self, pos_recon_corners: Tuple[Tuple[int]]):
        self.pos_recon_corners = pos_recon_corners
        self.recon_rectangle_selected = True

class Placement:
    """
    In each Position several shifted images are taken. These are called placements and informations are shared here.
    Order: Experment -> Position -> Placement -> Timestep
    """
    def __init__(
            self,
            place_dir: Union[str, Path],
            position: Position,
            ):
        """
        

        Parameters
        ----------
        place_dir : Union[str, Path]
            path to the placement directory eg. ...\2023-03-02 11-30-06\00001\00001_00005
        position : Position
            Position class with all the relevant informations about the position.

        Returns
        -------
        None.

        """
    
        self.place_dir: Path = Path(place_dir)
        "path to the placement directory"
        self.position = position
        "Position class with all the relevant informations about the position."
        self.place_name: str = place_dir.name
        "name of the placement eg. 00001_00005"
        self.place_image_roi: Mask = None
        "placment roi mask is the positions roi mask plus the placements shift from previous timestep"
        self.place_recon_corners: Tuple[Tuple[int]] = self.position.get_pos_recon_corners()
        "placment reconstruction corners are the positions reconstruction corners plus the placements shift from previous timestep"
        self.shift_vector: Tuple[int] = (0,0)
        "shif relative to the first placement, calculated with phase cross correlation"
        self.x0_guess: float = cfg.reconstruction_distance_guess
        "guess for the focusing distance. Is updated each timestep with the last focusing distance"
        self.X_plane: npt.NDArray[np.float64] = None
        "X_plane of the full image (rows: pixels, columns: polynomial features of pixel coordinates)"
        self.X_plane_pseudoinverse: npt.NDArray[np.float64] = None
        "Moore Penrose Pseudoinverse of self.X_plane"
        self.X_plane_image_roi: npt.NDArray[np.float64] = None
        "X_plane of the roi image (rows: pixels, columns: polynomial features of pixel coordinates)"
        self.X_plane_image_roi_pseudoinverse: npt.NDArray[np.float64] = None
        "Moore Penrose Pseudoinverse of self.X_plane_image_roi"
        self.X_plane_recon_rectangle: npt.NDArray[np.float64] = None
        "X_plane of the reconstruction rectangle (rows: pixels, columns: polynomial features of pixel coordinates)"
        self.X_plane_recon_rectangle_pseudoinverse: npt.NDArray[np.float64] = None
        "Moore Penrose Pseudoinverse of self.X_plane_recon_rectangle"
        self.set_place_image_roi()
        self._calculate_X_planes()
        self._calculate_X_planes_pseudoinverse()
        self._calculate_X_plane_recon_rectangle()
        self._calculate_X_plane_recon_rectangle_pseudoinverse()
    
    def _calculate_X_planes(self):
        ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
        ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
        "calculates self.X_plane and self.X_plane_image_roi"
        X1, X2 = np.mgrid[:cfg.image_size[0], :cfg.image_size[1]]
        
        if self.X_plane is None:
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            self.X_plane = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X)
        
        if self.position.image_roi_selected:
            X1_image_roi, X2_image_roi = X1[self.place_image_roi==True], X2[self.place_image_roi==True]
            X_image_roi = np.hstack((X1_image_roi.reshape(-1,1) , X2_image_roi.reshape(-1,1)))
            self.X_plane_image_roi = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X_image_roi)
        else: 
            self.X_plane_image_roi = self.X_plane
    
    def _calculate_X_plane_recon_rectangle(self):
        "if X_plane of reconstruction rectangle, if is selected, else copy self.X_plane_image_roi."
        if self.position.recon_rectangle_selected:
            recon_corners = self.get_place_recon_corners()
            heigth = recon_corners[0][1]-recon_corners[0][0]
            width = recon_corners[1][1]-recon_corners[1][0]
            X1, X2 = np.mgrid[:heigth, :width]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            self.X_plane_recon_rectangle = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X)
        else:
            self.X_plane_recon_rectangle = self.X_plane_image_roi
            self._calculate_X_plane_recon_rectangle_pseudoinverse()
    
    def _calculate_X_planes_pseudoinverse(self):
        "calculates self.X_plane_pseudoinverse and self.X_plane_image_roi_pseudoinverse"
        if self.X_plane_pseudoinverse is None:
            self.X_plane_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())
        if self.position.image_roi_selected or self.X_plane_image_roi_pseudoinverse is None:
            self.X_plane_image_roi_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane_image_roi.transpose(), self.X_plane_image_roi)), self.X_plane_image_roi.transpose())
    
    def _calculate_X_plane_recon_rectangle_pseudoinverse(self):
        self.X_plane_recon_rectangle_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane_recon_rectangle.transpose(), self.X_plane_recon_rectangle)), self.X_plane_recon_rectangle.transpose())
        
    def get_place_image_roi(self) -> Mask:
        return self.place_image_roi.copy()
    
    def get_place_recon_corners(self) -> Tuple[Tuple[int]]:
        return self.place_recon_corners
    
    def get_shift_vector(self) -> Tuple[int]:
        return self.shift_vector
    
    def get_x0_guess(self) -> float:
        return self.x0_guess
    
    def get_X_plane(self) -> npt.NDArray[np.float64]:
        return self.X_plane
    
    def get_X_plane_image_roi(self) -> npt.NDArray[np.float64]:
        return self.X_plane_image_roi
    
    def get_X_plane_recon_rectangle(self) -> npt.NDArray[np.float64]:
        return self.X_plane_recon_rectangle
    
    def get_X_plane_image_roi_pseudoinverse(self) -> npt.NDArray[np.float64]:
        return self.X_plane_image_roi_pseudoinverse
    
    def get_X_plane_recon_rectangle_pseudoinverse(self) -> npt.NDArray[np.float64]:
        return self.X_plane_recon_rectangle_pseudoinverse
    
    def set_place_image_roi(self):
        self.place_image_roi = ndimage.shift(self.position.get_pos_image_roi(), shift=self.shift_vector, mode='wrap')
        
    def set_place_recon_corners(self):
        "Readjusts recon corners according to the self.shift_vector. Reconstruction rectangle is not pushed over the outside edges"
        pos_corners = self.position.get_pos_recon_corners()
        height = pos_corners[0][1] - pos_corners[0][0]
        width = pos_corners[1][1] - pos_corners[1][0]
        ((miny, maxy), (minx, maxx)) = self.get_place_recon_corners()
        if pos_corners[0][0] + np.round(self.get_shift_vector()[0])<0:
            miny = 0
            maxy = height
        if pos_corners[0][0] + np.round(self.get_shift_vector()[0])>cfg.image_size[0]:
            miny = cfg.image_size[0] - height
            maxy = cfg.image_size[0]
        if pos_corners[0][0] + np.round(self.get_shift_vector()[1])<0:
            minx = 0
            maxx = width
        if pos_corners[0][0] + np.round(self.get_shift_vector()[1])>cfg.image_size[1]:
            minx = cfg.image_size[1] - width
            maxx = cfg.image_size[1]
        self.place_recon_corners = ((miny, maxy), (minx, maxx))
    
    def set_shift_vector(self, shift_vector: Tuple[int]):
        "sets shift vector and looks if roi or reconstruction recatangle are selected. If so they are also adjusted"
        self.shift_vector = shift_vector
        if self.position.image_roi_selected or self.position.recon_rectangle_selected:
            self.set_place_image_roi()
            self.set_place_recon_corners()
            self._calculate_X_planes()
            self._calculate_X_planes_pseudoinverse()
        
    def set_x0_guess(self, x0_guess: float):
        self.x0_guess = x0_guess
    
        
class Hologram:
    """
        Class handles the reconstruction of the hologram. 
        It interacts with Koala, can find the true focus distance and saves the complex image at the focus distance.
    """
    def __init__(self,
                 fname: Union[str, Path],
                 placement: Placement,
                 focus: float = None,
                 ):
        """
        Parameters
        ----------
        fname : Union[str, Path]
            Path to the holograms .tif file.
        placement : Placement
            Class with informations about the placement of the hologram.
        focus : float, optional
            Focus distance, can be found with self.calculate_focus. The default is None.

        Returns
        -------
        None.

        """

        self.fname: Path = Path(fname)
        "Path to the holograms .tif file"
        self.corrupted = self._check_corrupted()
        "If the image is corruped in any way, this is turned to true. This hologram is not considered in the further workflow anymore."
        self.placement: Placement = placement
        "Class with informations about the placement"
        self.focus: float = focus 
        "Focus distance, can be found with self.calculate_focus. The default is None."
        self.focus_score: float = None
        "score of the focus cost function"
        self.nfev: int = 0
        "number of function evaluations. Very high number of evaluations indicate hard cost function, which indicate corrupted image"
        self.cplx_image: CplxImage = None # as soon as the focus point is found this function is evaluated
        "complex image at the focus distance, is added when calculating the focus."
    
    def calculate_focus(self):
        "calculates the focus either with local grid search or with scipy minimization algorithms. local grid search is recommended"
        cfg.KOALA_HOST.LoadHolo(str(self.fname),1)
        cfg.KOALA_HOST.SetUnwrap2DState(True)
        
        if cfg.local_grid_search:
            "starting search grid"
            xmin, xmax = cfg.reconstruction_distance_low, cfg.reconstruction_distance_high
            for i in range(len(cfg.nfevaluations)):
                x = np.linspace(xmin, xmax, cfg.nfevaluations[i])
                focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                while np.argmin(focus_scores) == 0 and self.nfev<cfg.nfev_max: # if first element is the minimum, appending grid is tested
                    x = np.linspace(xmin-(xmax-xmin), xmin, cfg.nfevaluations[i])
                    xmin, xmax = x[0], x[-1]
                    focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                while np.argmin(focus_scores) == len(x)-1 and self.nfev<cfg.nfev_max: # if last element is the minimum, appending grid is tested
                    x = np.linspace(xmax, xmax+(xmax-xmin), cfg.nfevaluations[i])
                    xmin, xmax = x[0], x[-1]
                    focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                "adjusting search grid"
                spacing = x[1] - x[0]
                xmin = x[np.argmin(focus_scores)] - spacing/2
                xmax = x[np.argmin(focus_scores)] + spacing/2
                if cfg.nfev_max<self.nfev:
                    print(f'{self.fname} could not find a focus point')
                    self.corrupted = True
            self.focus = x[np.argmin(focus_scores)]
            self.focus_score = np.min(focus_scores)
            if self.focus<cfg.reconstruction_distance_low or cfg.reconstruction_distance_high<self.focus:
                print(f'{self.fname} focus is out of borders with {np.round(self.focus,3)}')
                self.corrupted = True
        else:
            "scipy minimization"
            bounds = Bounds(lb=cfg.reconstruction_distance_low, ub=cfg.reconstruction_distance_high)
            res = minimize(self._evaluate_reconstruction_distance, [self.placement.get_x0_guess()], method=cfg.optimizing_method, bounds=bounds)
            self.focus = res.x[0]
            self.placement.set_x0_guess(self.focus)
            self.focus_score = res.fun
            self.nfev = res.nfev
        self.cplx_image = self._cplx_image()
        
    def _check_corrupted(self):
        "There are images that are only black, those images have a small size"
        threshold = 1e5
        size = os.path.getsize(self.fname)
        if size < threshold:
            print(f'image {self.fname} is corrupted!')
            return True
        else:
            return False
            
    def _cplx_image(self) -> CplxImage:
        "calculates complex image at focus distance with the help of Koala"
        cfg.KOALA_HOST.LoadHolo(str(self.fname),1)
        cfg.KOALA_HOST.SetUnwrap2DState(True)
        cfg.KOALA_HOST.SetRecDistCM(self.focus)
        cfg.KOALA_HOST.OnDistanceChange()
        ph = cfg.KOALA_HOST.GetPhase32fImage()
        ph = self._subtract_plane(ph)
        if cfg.use_amp:
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            cplx_image = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx_image = np.exp(complex(0.,1.)*ph)
        return cplx_image.astype(np.complex64)
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance: List[float], focus_method_nr: int = 0) -> float:
        """returns the score of the chosen focus method. reconstruction_distance is a list with the reconstruction distance to work with the
        scypi minimization function. If a scypi minimization function and not local grid search first focus method of cfg.focus_method is used."""
        self.nfev += 1
        cfg.KOALA_HOST.SetRecDistCM(reconstruction_distance[0])
        cfg.KOALA_HOST.OnDistanceChange()
        if cfg.focus_method[focus_method_nr] == 'std_amp':
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            amp = self._subtract_plane_recon_rectangle(amp)
            return np.std(amp)
        elif cfg.focus_method[focus_method_nr] == 'phase_sharpness':
            ph = cfg.KOALA_HOST.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -self._evaluate_phase_sharpness(ph)
        elif cfg.focus_method[focus_method_nr] == 'combined':
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            amp = self._subtract_plane_recon_rectangle(amp)
            ph = cfg.KOALA_HOST.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -np.std(ph)/np.std(amp)
        else:
            print("Method ", cfg.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_phase_sharpness(self, gray_image) -> float:
        "claculates the sharpness of the input image"
        gray_image = gray_image.clip(min=0)
        # Calculate gradient magnitude using Sobel filter
        grad_x = ndimage.sobel(gray_image, axis=0)
        grad_y = ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 4 + grad_y ** 4)
    
    def get_cplx_image(self) -> CplxImage:
        if self.cplx_image is None:
            if self.focus is None:
                self.calculate_focus()
            self.cplx_image = self._cplx_image()
        return self.cplx_image.copy()
    
    def _subtract_plane(self, field: Image) -> CplxImage:
        "subtracts the surface plane for the whole image with linear regression."
        place_mask = self.placement.get_place_image_roi()
        theta = np.dot(self.placement.get_X_plane_image_roi_pseudoinverse(), field[place_mask==True].reshape(-1))
        plane = np.dot(self.placement.get_X_plane(), theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _subtract_plane_recon_rectangle(self, field: Image) -> CplxImage:
        "subtracts the surface plane for the for the reconstruction part of the image with linear regression."
        field = crop_image(field, self.placement.get_place_recon_corners())
        theta = np.dot(self.placement.get_X_plane_recon_rectangle_pseudoinverse(), field.reshape(-1))
        ymin, ymax = self.placement.get_place_recon_corners()[0][0], self.placement.get_place_recon_corners()[0][1]
        plane = np.dot(self.placement.get_X_plane_recon_rectangle(), theta).reshape((ymax-ymin)%(cfg.image_size[0]+1), -1)
        return field-plane

class SpatialPhaseAveraging:
    """
        Class subtracts the background and spatially averages a set of overlapping images.
    """
    def __init__(self,
                 position: Position,
                 placements: List[Placement],
                 timestep: int,
                 ):
        """
        Parameters
        ----------
        position : Position
            Position class with all the relevant informations about the position.
        placements : List[Placement]
            List of all placements. Placement is a class with informations about the placement of the hologram.
        timestep : int
            Timestep of the images which are averaged.
            
        Returns
        -------
        None.

        """
        self.position: Position = position
        "Position class with all the relevant informations about the position"
        self.timestep: int = timestep
        "timestep of the images which are averaged"
        self.placements: List[Placement] = placements
        "list of all placements. Placement is a class with informations about the placement of the hologram"
        self.holograms: List[Hologram] = self._generate_holograms()
        "list with the hologrmas class for each of the placements at this timestep."
        self.num_place: int = len(self.holograms)
        "number of holograms, which are in use (corruped holograms are not in use)."
        self.background: CplxImage = self._background()
        "background of the images"
        self.spatial_avg: CplxImage = self._spatial_avg()
        "spatial average of all the holograms"
    
    
    def _background(self) -> CplxImage:
        "if position has a background this is taken. Else the background of this timestep is calculated. Appends this background to the position."
        if self.position.background is not None:
            return self.position.get_background()
        else:
            images = np.zeros((len(self.holograms), cfg.image_size[0], cfg.image_size[1]), dtype=np.complex64)
            for i in range(len(self.holograms)):
                images[i] = self.holograms[i].get_cplx_image()
            background = (np.median(np.abs(images), axis=0)*np.exp(1j*np.median(np.angle(images), axis=0))).astype(np.complex64)
            
            # Averaging the first 10 backgrounds
            self.position.backgrounds.append(background)
            if len(self.position.backgrounds) == 10:
                self.position.average_backgrounds()
            
            return background
    
    def _spatial_avg(self) -> CplxImage:
        "spatial average of all the images. (repeat for each image(Image -> background subtracted ->shifted->added))/number of images"
        spatial_avg = self.holograms[0].get_cplx_image()
        spatial_avg /= self.background
        place0_roi = self.placements[0].get_place_image_roi()
        place0_recon_corners = self.placements[0].get_place_recon_corners()
        for i in range(1, self.num_place):
            cplx_image = self.holograms[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image, shift_vector = self._shift_image(spatial_avg, cplx_image, place0_recon_corners)
            self.placements[i].set_shift_vector(shift_vector)
            cplx_image = self._subtract_phase_offset(spatial_avg, cplx_image, place0_roi)
            spatial_avg += cplx_image
        return spatial_avg/self.num_place
    
    def _generate_holograms(self) -> List[Hologram]:
        "generates the holograms of all placements and calculates their focus. Corruped holograms are rejected."
        holograms = []
        for pl in self.placements:
            fname = Path(str(pl.place_dir) + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, pl)
            if hologram.corrupted:
                continue
            hologram.calculate_focus()
            # first guess is the focus point of the last image
            pl.set_x0_guess(hologram.focus)
            holograms.append(hologram)
        return [h for h in holograms if not h.corrupted]
    
    def get_background(self) -> CplxImage:
        return self.background.copy()
    
    def get_spatial_avg(self) -> CplxImage:
        return self.spatial_avg.copy()
    
    def _shift_image(self, reference_image: CplxImage, moving_image: CplxImage, corners: Tuple[int]) -> (CplxImage, Tuple[int]):
        "calculates the shift between tow images. The phase image is calculated only on the reconstruction recangle."
        ref = np.angle(reference_image[corners[0][0]:corners[0][1], corners[1][0]:corners[1][1]])
        mov = np.angle(moving_image[corners[0][0]:corners[0][1], corners[1][0]:corners[1][1]])
        # increase in importance to the higher areas (bacteria)
        ref = np.exp(5*ref)
        mov = np.exp(5*mov)
        try: # from scikit-image version 0.19.1 they added normalization. base configuration is 'phase', but None works better
            shift_measured, _, __ = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None, return_error='always')
        except TypeError: # Invalid argument normalization
            shift_measured, _, __ = phase_cross_correlation(ref, mov, upsample_factor=10, return_error=True)
        shift_vector = (shift_measured[0], shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shift_vector, mode='constant')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shift_vector, mode='constant')
        shift_vector = (int(np.round(shift_measured[0],0)), int(np.round(shift_measured[1],0)))
        return real+complex(0.,1.)*imaginary, shift_vector
        
    def _subtract_phase_offset(self, avg: CplxImage, new: CplxImage, mask: Mask) -> CplxImage:
        "aligns the phases of the different iamges."
        z= np.angle(np.multiply(new[mask==True],np.conj(avg[mask==True]))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new


class Pipeline:
    """
        The pipeline calculates and saves the clean phase images of the time-lapse for all selected positions
    """
    def __init__(
            self,
            base_dir: Union[str, Path],
            saving_dir: Union[str, Path] = None,
            restrict_positions: slice = None,
            restrict_timesteps: range = None,
            ):
        """
        Parameters
        ----------
        base_dir : Union[str, Path]
            Path to the directory where the experiment is saved.
        saving_dir : Union[str, Path], optional
            Path to the directory where the processed images should be saved. The default is None.
        restrict_positions : slice, optional
            Slice of the positions that should be processed. The default is None.
        restrict_timesteps : range, optional
            Range of the timesteps that should be processed. The default is None.
            
        Returns
        -------
        None.

        """
        self.base_dir: Path = Path(base_dir)
        "Path to the directory where the experiment is saved"
        self.saving_dir: Path = self._saving_dir(saving_dir)
        "Path to the directory where the processed images should be saved"
        self.data_file_path: Path = None
        "Path where the information about the processing is saved"
        self.restrict_positions: slice = restrict_positions
        "Slice of the positions that should be processed"
        self.restrict_timesteps: range = restrict_timesteps
        "Range of the timesteps that should be processed"
        self.positions: List[Position] = self._positions()
        "List of the positions that are processed"
        self.first_timestep: int = None
        "number of the first timestep in the time-lapse (normally 0 or 1)"
        self.timesteps: range = self._timesteps()
        "Range of the timesteps that are processed"
        self.image_settings_updated: bool = False
        "Checks if the image settings in the config files are updated"
        self.image_count: int = 0
        "count of the processed images, used for periodically restarting Koala"
        start_koala()

    
    def get_bf_image(self, phase_image: Image, t: int) -> Image:
        bf_fname = 0
        bf = tifffile.imread(bf_fname)[512:1536,512:1536]
        bf = np.fliplr(bf)
        bf = trans.rotate(bf, -90, mode="edge")
        ph = np.zeros(bf.shape)
        ph[:phase_image.shape[0], :phase_image.shape[1]] = phase_image
        bf_ = gradient_squared(bf)
        ph_ = gradient_squared(ph)
        rot, zoomlevel = grid_search_2d(ph_, bf_, cfg.bf_rot_guess, cfg.bf_zoom_guess, cfg.bf_rot_search_length,
                                        cfg.bf_zoom_search_length, cfg.bf_local_searches)
        cfg.set_bf_rot_zoom(rot, zoomlevel)
        bf_rz = zoom(trans.rotate(bf_, rot, mode="edge"),zoomlevel).astype(np.float32)
        return bf_rz[:phase_image.shape[0], :phase_image.shape[1]]
    
    
    def _get_mask_from_rectangle(self, image: Image, title: str = None) -> Mask:
        "shows an image and waits until a recangle is selected. Returns the mask of the rectangle"
        # Show the image and wait for user to select a rectangle
        if title is None:
            title = "Select region of interest"
        cv2.imshow(title, image)
        rect = cv2.selectROI(title, image, False)
        cv2.destroyAllWindows()
        
        # Create a mask with the same shape as the image, initialized to zeros
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Set the pixels inside the rectangle to 1 in the mask
        x, y, w, h = rect
        mask[y:y+h, x:x+w] = 1
        
        return mask
        
    
    def _get_rectangle_coordinates(self, image: Image, title: str = None) -> Tuple[Tuple[int]]:
        "shows an image and waits until a recangle is selected. Returns the corners of the rectangle"
        # Show the image and wait for user to select a rectangle
        if title is None:
            title = "Select reconstruction rectangle"
        cv2.imshow(title, image)
        rect = cv2.selectROI(title, image, False)
        cv2.destroyAllWindows()
    
        # Extract coordinates of the rectangle
        x, y, w, h = rect
        ymin, ymax = y, y + h
        xmin, xmax = x, x + w
        
        return ((ymin, ymax), (xmin, xmax))

    def _positions(self) -> List[Position]:
        "returns a list of the selected positions"
        all_positions =[Position(Path(f.path)) for f in os.scandir(self.base_dir) if f.is_dir()]
        if self.restrict_positions == None:
            return all_positions
        else:
            return all_positions[self.restrict_positions]
        
    def _positions_image_roi_corners(self, po) -> Tuple[Tuple[int]]:
        "returns the corners of the region of interest"
        if po.image_roi_selected:
            return get_masks_corners(po.pos_image_roi)
        else:
            return cfg.image_cut
    
    def process(self):
        "processes the images. First loop is through the positions, secand through the timesteps."
        if cfg._LOADED is False:
            raise RuntimeError(
                "configuration has not been loaded, do so by executing sa.config.load_config"
            )
        for po in self.positions:
            cfg.image_cut = self._positions_image_roi_corners(po)
            placements = [Placement(place_dir=Path(str(f.path)), position=po) for f in os.scandir(str(po.pos_dir)) if f.is_dir()] # list of all placements
            last_phase_image = None
            for t in self.timesteps:
                start_image = time.time()
                spa = SpatialPhaseAveraging(po, placements, t) 
                averaged_phase_image = get_result_unwrap(np.angle(spa.get_spatial_avg())).astype(np.float32)
                if last_phase_image is not None:
                    averaged_phase_image = self._temporal_shift_correction(last_phase_image, averaged_phase_image)
                
                if not self.image_settings_updated:
                    cfg.set_image_variables((cfg.KOALA_HOST.GetPhaseWidth(),cfg.KOALA_HOST.GetPhaseHeight()), cfg.KOALA_HOST.GetPxSizeUm()*1e-6, cfg.KOALA_HOST.GetLambdaNm(0)*1e-9)
                    self._update_image_cut(spa)
                    self.data_file_path = self._write_data_file()
                    self.image_settings_updated = True
                
                phase_image =  averaged_phase_image[cfg.image_cut[0][0]:cfg.image_cut[0][1],cfg.image_cut[1][0]:cfg.image_cut[1][1]]
                
                save_pos_folder = str(self.saving_dir) + os.sep + po.pos_name
                if not os.path.exists(save_pos_folder):
                    os.makedirs(save_pos_folder)
                
                self._save_image(phase_image, save_pos_folder, t)
                
                last_phase_image = averaged_phase_image
                duration_timestep = np.round(time.time()-start_image,1)
                print(f"pos: {po.pos_name}, timestep: {t} done in {duration_timestep} seconds")
                self._update_data_file(spa, duration_timestep)
                
                self.image_count += 1
                if self.image_count % cfg.koala_reset_frequency == 0:
                    if cfg.display_always_on:
                        shut_down_restart_koala()
                    else:
                        logout_login_koala()
                del spa
                
            del placements
            gc.collect()
            
    def _save_image(self, phase_image, save_pos_folder, t):
        "saves the images in the selected format. If .tif is selected it is saved twice to conform with the delta pipeline"
        if cfg.save_format == ".delta":
            ph_scaled = (((phase_image + np.pi/2) / np.pi) * 65535).astype(np.uint16)
            fname_scaled = save_pos_folder + os.sep + f"pos{save_pos_folder[-5:]}cha1fra{str(t + 1 - self.first_timestep).zfill(5)}.tif"
            fname_scaled2 = save_pos_folder + os.sep + f"pos{save_pos_folder[-5:]}cha2fra{str(t + 1 - self.first_timestep).zfill(5)}.tif"
            if cfg.bf_image:
                bf = self.calculate_bf_image(phase_image, t)
                bf_scaled = ((bf - bf.min())/(bf.max() - bf.min()) * 65535).astype(np.uint16)
                tifffile.imwrite(fname_scaled, bf_scaled)
            else:
                tifffile.imwrite(fname_scaled, ph_scaled)
            tifffile.imwrite(fname_scaled2, ph_scaled)
        
        if cfg.save_format == ".tif":
            ph_fname = save_pos_folder +"\\ph_timestep_" + str(t).zfill(5) + cfg.save_format
            tifffile.imwrite(ph_fname, phase_image)
            if cfg.bf_image:
                bf_fname = save_pos_folder +"\\bf_timestep_" + str(t).zfill(5) + cfg.save_format
                tifffile.imwrite(bf_fname, self.calculate_bf_image(phase_image, t))
            
        if cfg.save_format == ".bin":
            fname = save_pos_folder +"\\ph_timestep_" + str(t).zfill(5) + cfg.save_format
            binkoala.write_mat_bin(fname, phase_image, phase_image.shape[0], phase_image.shape[1], cfg.px_size, cfg.hconv, cfg.unit_code)
            if cfg.bf_image:
                bf_fname = save_pos_folder +"\\bf_timestep_" + str(t).zfill(5) + '.tif'
                tifffile.imwrite(bf_fname, self.calculate_bf_image(phase_image, t))
    
    def _saving_dir(self, saving_dir: Union[str, Path]) -> Path:
        "returns the saveing dir"
        if saving_dir == None:
            saving_dir = Path(str(self.base_dir) + " phase averages")
            if not os.path.exists(str(saving_dir)):
                os.makedirs(str(saving_dir))
        return Path(saving_dir)
    
    def select_positions_image_roi(self, same_for_all_pos = False):
        "allows for selection of the region of interest, for each position."
        mask = None
        for po in self.positions:
            p0_dir = Path(str(po.pos_dir) + os.sep + [d for d in os.listdir(str(po.pos_dir)) if os.path.isdir(Path(po.pos_dir,d))][0])
            p0 = Placement(place_dir=p0_dir, position=po)
            fname = Path(str(p0.place_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p0, focus = cfg.reconstruction_distance_guess)
            ph_image = np.angle(hologram.get_cplx_image())
            if mask is None or not same_for_all_pos:
                mask = self._get_mask_from_rectangle(ph_image)
            po.set_pos_image_roi(mask)
    
    def select_positions_recon_rectangle(self, same_for_all_pos = False):
        "allows for selection of the reconstruction rectangle (part where the focusing mehtod is applied), for each position."
        crop_coords = None
        for po in self.positions:
            p0_dir = Path(str(po.pos_dir) + os.sep + [d for d in os.listdir(str(po.pos_dir)) if os.path.isdir(Path(po.pos_dir,d))][0])
            p0 = Placement(place_dir=p0_dir, position=po)
            fname = Path(str(p0.place_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p0, focus = cfg.reconstruction_distance_guess)
            ph_image = np.angle(hologram.get_cplx_image())
            if crop_coords is None or not same_for_all_pos:
                crop_coords = self._get_rectangle_coordinates(ph_image)
            po.set_pos_recon_corners(crop_coords)
            
    def _temporal_shift_correction(self, reference_image: Image, moving_image: Image) -> Image:
        "images can move overtime, this function corrects for the shift. Movement is due to different focus distances"
        try: # from scikit-image version 0.19.1 they added normalization. base configuration is 'phase', but None works better
            shift_measured, _, __ = phase_cross_correlation(reference_image, moving_image, upsample_factor=10, normalization=None, return_error='always')
        except TypeError: # Invalid argument normalization
            shift_measured, _, __ = phase_cross_correlation(reference_image, moving_image, upsample_factor=10, return_error=True)
        shift_vector = (shift_measured[0],shift_measured[1])
        return ndimage.shift(moving_image, shift=shift_vector, mode='constant')
        
    def _timesteps(self) -> range:
        "returns the range of the timesteps processed"
        holo_path = str(self.base_dir)+os.sep+self.positions[0].pos_name + os.sep + "00001_00001\Holograms"
        self.first_timestep = int(sorted(os.listdir(holo_path))[0][:5])
        if self.restrict_timesteps == None:
            num_timesteps = len(os.listdir(holo_path))
            all_timesteps = range(self.first_timestep, self.first_timestep + num_timesteps)
            return all_timesteps
        else:
            return self.restrict_timesteps
    
    def _update_data_file(self, spa, time):
        "updates the data file with informations about the current averaged image"
        with open(self.data_file_path, 'r') as file:
            data = json.load(file)
        
        position = spa.position.pos_name
        timestep = spa.timestep
        image_name = f'position_{position}_timestep_{str(timestep).zfill(5)}'
        data["images"][image_name] = {
            "position" : int(position),
            "timestep" : timestep,
            "time": time,
            "foci": tuple(holo.focus for holo in spa.holograms),
            "function_evaluations": int(np.sum([holo.nfev for holo in spa.holograms])),
            "shift_x": tuple(int(placement.get_shift_vector()[1]) for placement in spa.placements),
            "shift_y": tuple(int(placement.get_shift_vector()[0]) for placement in spa.placements),
        }
        
        with open(self.data_file_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    def _update_image_cut(self, spa: SpatialPhaseAveraging):
        "shifts the image cut to account for temporal movement"
        y_shifts = np.array([placement.get_shift_vector()[0] for placement in spa.placements])
        x_shifts = np.array([placement.get_shift_vector()[1] for placement in spa.placements])
        y_midpoint = np.mean([np.min(y_shifts), np.max(y_shifts)])
        x_midpoint = np.mean([np.min(x_shifts), np.max(x_shifts)])
        image_cut = ((50+int(y_midpoint), 750+int(y_midpoint)), (50+int(x_midpoint), 750+int(x_midpoint)))
        cfg.set_image_cut(image_cut)

    def _write_data_file(self) -> Path:
        "writes an data file with the informations about the processing"
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        data_file_path = Path(self.saving_dir, f'data file {current_datetime}.json')
        
        data = {
            "settings": {
                "base_dir": str(self.base_dir),
                "koala_configuration": cfg.koala_config_nr,
                "focus_method": cfg.focus_method,
                "optimizing_method": cfg.optimizing_method,
                "local_grid_search": cfg.local_grid_search,
                "nfevaluations": cfg.nfevaluations,
                "final_grid_spacing": (cfg.reconstruction_distance_high-cfg.reconstruction_distance_low)/np.prod([f-1 for f in cfg.nfevaluations]),
                "nfev_max": cfg.nfev_max,
                "tolerance": cfg.tolerance,
                "reconstruction_distance_low": cfg.reconstruction_distance_low,
                "reconstruction_distance_high": cfg.reconstruction_distance_high,
                "reconstruction_distance_guess": cfg.reconstruction_distance_guess,
                "plane_fit_order": cfg.plane_fit_order,
                "use_amp": cfg.use_amp,
                "image_size": cfg.image_size,
                "px_size": cfg.px_size,
                "hconv": cfg.hconv,
                "unit_code": cfg.unit_code,
                "image_cut": cfg.image_cut,
                "save_format": cfg.save_format,
                "koala_reset_frequency": cfg.koala_reset_frequency,
                "restart_koala": cfg.display_always_on,
            },
            "images": {}
        }
        
        if self.positions[0].image_roi_selected:
            for pos in self.positions:
                data["settings"][f'image_cut_pos{pos.pos_name}'] = self._positions_image_roi_corners(pos)
            del(data["settings"]["image_cut"])
        
        with open(data_file_path, 'w') as file:
            json.dump(data, file, indent=4)  # Add indent parameter to make it readable
        
        return data_file_path
        