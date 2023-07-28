# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:24:04 2023

@author: SWW-Bc20
"""
import os
import numpy as np
import gc
import time
import numpy.typing as npt
import cv2
import tifffile
import json
from datetime import datetime
from typing import cast, List, Union, Dict, Optional, Any, Tuple
from pathlib import Path
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

from .utilities import cfg, get_result_unwrap, logout_login_koala, shut_down_restart_koala, crop_image
from . import binkoala

Image = npt.NDArray[np.float32]

CplxImage = npt.NDArray[np.complex64]

Mask = npt.NDArray[np.uint8]


class Position:
    def __init__(
            self,
            pos_dir: Union[str, Path],
            ):
        
        self.pos_dir: Path = Path(pos_dir)
        self.pos_name: str = pos_dir.name
        self.image_roi_selected: bool = False
        self.pos_image_roi: Mask = np.ones(cfg.image_size, dtype=np.uint8)
        self.recon_rectangle_selected: bool = False
        self.pos_recon_corners: List[List[int]] = None  #((ymin, ymax), (xmin, xmax))
        self.backgrounds = []
        self.background = self._background()
        
    def average_backgrounds(self):
        self.background = np.mean(np.abs(self.backgrounds), axis=0) * np.exp(1j* np.mean(np.angle(self.backgrounds), axis=0)).astype(np.complex64)
        tifffile.imwrite(str(self.pos_dir)+os.sep+'background.tif', self.background)
    
    def _background(self):
        # checks whether there is a background already calculated for this Position
        if os.path.isfile(str(self.pos_dir)+os.sep+'background.tif'):
            return np.array(tifffile.imread(str(self.pos_dir)+os.sep+'background.tif'))
        else:
            return None
            
    def get_background(self):
        return self.background.copy()
    
    def get_pos_image_roi(self) -> Mask:
        return self.pos_image_roi.copy()
    
    def get_pos_image_roi_corners(self) -> List[List[int]]:
        if self.image_roi_selected:
            non_zero_indices = np.nonzero(self.pos_image_roi)
            [ymin, ymax] = [np.min(non_zero_indices[0]), np.max(non_zero_indices[0])]
            [xmin, xmax] = [np.min(non_zero_indices[1]), np.max(non_zero_indices[1])]
            return [[ymin, ymax], [xmin, xmax]]
        else:
            return [[0,cfg.image_size[0]],[0,cfg.image_size[1]]]
    
    def set_pos_image_roi(self, pos_image_roi: Mask):
        self.pos_image_roi = pos_image_roi.astype(np.uint8)
        self.image_roi_selected = True
    
    def get_pos_recon_corners(self) -> List[List[int]] :
        if self.pos_recon_corners is None:
            return self.get_pos_image_roi_corners()
        else:
            return self.pos_recon_corners
    
    def set_pos_recon_corners(self, pos_recon_corners: List[List[int]]):
        self.pos_recon_corners = pos_recon_corners
        self.recon_rectangle_selected = True


class Placement:
    def __init__(
            self,
            place_dir: Union[str, Path],
            position: Position,
            ):
    
        self.place_dir: Path = Path(place_dir)
        self.position = position
        self.place_name: str = place_dir.name
        self.place_image_roi: Mask = None
        self.place_recon_corners: List[List[int]] = self.position.get_pos_image_roi_corners()
        self.shift_vector: List[int] = [0,0]
        self.x0_guess: float = cfg.reconstruction_distance_guess
        self.X_plane: npt.NDArray[np.float64] = None
        self.X_plane_pseudoinverse: npt.NDArray[np.float64] = None
        self.X_plane_image_roi: npt.NDArray[np.float64] = None
        self.X_plane_image_roi_pseudoinverse: npt.NDArray[np.float64] = None
        self.set_place_image_roi()
        self._calculate_X_planes()
        self._calculate_X_planes_pseudoinverse()
        # Before the shift_vector is estimated the full image is used for calculating the focus point
        self.first_timestep: bool = True
        self.X_plane_recon_rectangle: npt.NDArray[np.float64] = self.X_plane
        self.X_plane_recon_rectangle_pseudoinverse: npt.NDArray[np.float64] = self.X_plane_pseudoinverse
    
    def _calculate_X_planes(self):
        ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
        ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
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
        heigth = (self.place_recon_corners[0][1]-self.place_recon_corners[0][0]) % (cfg.image_size[0]+1)
        width = (self.place_recon_corners[1][1]-self.place_recon_corners[1][0]) % (cfg.image_size[0]+1)
        X1, X2 = np.mgrid[:heigth, :width]
        X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
        self.X_plane_recon_rectangle = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X)
    
    def _calculate_X_planes_pseudoinverse(self):
        if self.X_plane_pseudoinverse is None:
            self.X_plane_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())
        if self.position.image_roi_selected or self.X_plane_image_roi_pseudoinverse is None:
            self.X_plane_image_roi_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane_image_roi.transpose(), self.X_plane_image_roi)), self.X_plane_image_roi.transpose())
    
    def _calculate_X_plane_recon_rectangle_pseudoinverse(self):
        self.X_plane_recon_rectangle_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane_recon_rectangle.transpose(), self.X_plane_recon_rectangle)), self.X_plane_recon_rectangle.transpose())
        
    def get_place_image_roi(self) -> Mask:
        return self.place_image_roi.copy()
    
    def get_place_recon_corners(self) -> List[List[int]]:
        return self.place_recon_corners
    
    def get_shift_vector(self) -> List[int]:
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
        height = self.position.get_pos_recon_corners()[0][1] - self.position.get_pos_recon_corners()[0][0]
        width = self.position.get_pos_recon_corners()[1][1] - self.position.get_pos_recon_corners()[1][0]
        if self.position.get_pos_recon_corners()[0][0] + np.round(self.get_shift_vector()[0])<0:
            self.place_recon_corners[0][0] = 0
            self.place_recon_corners[0][1] = height
        if self.position.get_pos_recon_corners()[0][0] + np.round(self.get_shift_vector()[0])>cfg.image_size[0]:
            self.place_recon_corners[0][0] = cfg.image_size[0] - height
            self.place_recon_corners[0][1] = cfg.image_size[0]
        if self.position.get_pos_recon_corners()[0][0] + np.round(self.get_shift_vector()[1])<0:
            self.place_recon_corners[1][0] = 0
            self.place_recon_corners[1][1] = width
        if self.position.get_pos_recon_corners()[0][0] + np.round(self.get_shift_vector()[1])>cfg.image_size[1]:
            self.place_recon_corners[1][0] = cfg.image_size[1] - width
            self.place_recon_corners[1][1] = cfg.image_size[1]
    
    def set_shift_vector(self, shift_vector: List[int]):
        self.shift_vector = shift_vector
        if self.position.image_roi_selected or self.position.recon_rectangle_selected:
            self.set_place_image_roi()
            self.set_place_recon_corners()
            self._calculate_X_planes()
            self._calculate_X_planes_pseudoinverse()
        if self.first_timestep:
            self._calculate_X_plane_recon_rectangle()
            self._calculate_X_plane_recon_rectangle_pseudoinverse()
            self.first_timestep = False
        
    def set_x0_guess(self, x0_guess: float):
        self.x0_guess = x0_guess
    
        
class Hologram:
    def __init__(self,
                 fname: Union[str, Path],
                 placement: Placement,
                 focus: float = None,
                 ):

        self.fname: Path = Path(fname)
        self.corrupted = self._check_corrupted()
        self.placement: Placement = placement
        self.focus: float = focus # Focus distance
        self.focus_score: float = None # score of evaluatino function at the Focus point (minimum)
        self.nfev: int = 0 # number of function evaluatons needed
        self.cplx_image: CplxImage = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self):
        cfg.KOALA_HOST.LoadHolo(str(self.fname),1)
        cfg.KOALA_HOST.SetUnwrap2DState(True)
        
        if cfg.local_grid_search:
            xmin, xmax = cfg.reconstruction_distance_low, cfg.reconstruction_distance_high
            for i in range(len(cfg.focus_method)):
                x = np.linspace(xmin, xmax, cfg.nfevaluations[i])
                focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                while np.argmin(focus_scores) == 0 and cfg.nfev_max<self.nfev:
                    x = np.linspace(xmin-(xmax-xmin), xmin, cfg.nfevaluations[i])
                    focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                while np.argmin(focus_scores) == len(x)-1 and cfg.nfev_max<self.nfev:
                    x = np.linspace(xmax, xmax+(xmax-xmin), cfg.nfevaluations[i])
                    focus_scores = np.array([self._evaluate_reconstruction_distance([x[j]], i) for j in range(x.shape[0])])
                spacing = x[1] - x[0]
                xmin = x[np.argmin(focus_scores)] - spacing/2
                xmax = x[np.argmin(focus_scores)] + spacing/2
                if cfg.nfev_max<self.nfev:
                    print(f'{self.fname} could not find a focus point')
                    self.corrupted = True
            self.focus = x[np.argmin(focus_scores)]
            self.focus_score = np.min(focus_scores)
            if self.focus<cfg.reconstruction_distance_low or cfg.reconstruction_distance_high<self.focus:
                self.corrupted = True
                print(f'{self.fname} focus is out of borders with {np.round(self.focus,3)}')
        else:
            bounds = Bounds(lb=cfg.reconstruction_distance_low, ub=cfg.reconstruction_distance_high)
            res = minimize(self._evaluate_reconstruction_distance, [self.placement.get_x0_guess()], method=cfg.optimizing_method, bounds=bounds)
            self.focus = res.x[0]
            self.placement.set_x0_guess(self.focus)
            self.focus_score = res.fun
            self.nfev = res.nfev
        self.cplx_image = self._cplx_image()
        
    def _check_corrupted(self):
        # There are images that are only black, those images have a small size
        threshold = 1e5
        size = os.path.getsize(self.fname)
        if size < threshold:
            print(f'image {self.fname} is corrupted!')
            return True
        else:
            return False
            
    def _cplx_image(self) -> CplxImage:
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
        self.nfev += 1
        cfg.KOALA_HOST.SetRecDistCM(reconstruction_distance[0])
        cfg.KOALA_HOST.OnDistanceChange()
        if cfg.focus_method[focus_method_nr] == 'std_amp':
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            amp = self._subtract_plane_recon_rectangle(amp)
            return np.std(amp)
        elif cfg.focus_method[focus_method_nr] == 'std_ph_sobel':
            ph = cfg.KOALA_HOST.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -self._evaluate_std_ph_sobel(ph)
        elif cfg.focus_method[focus_method_nr] == 'combined':
            amp = cfg.KOALA_HOST.GetIntensity32fImage()
            amp = self._subtract_plane_recon_rectangle(amp)
            ph = cfg.KOALA_HOST.GetPhase32fImage()
            ph = self._subtract_plane_recon_rectangle(ph)
            return -np.std(ph)/np.std(amp)
        else:
            print("Method ", cfg.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_std_ph_sobel(self, gray_image) -> float:
        gray_image = gray_image.clip(min=0)
        gray_image = cv2.resize(gray_image, (gray_image.shape[0]//2, gray_image.shape[1]//2), interpolation = cv2.INTER_AREA)
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
        theta = np.dot(self.placement.get_X_plane_image_roi_pseudoinverse(), field[self.placement.get_place_image_roi()==True].reshape(-1))
        plane = np.dot(self.placement.get_X_plane(), theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _subtract_plane_recon_rectangle(self, field: Image) -> CplxImage:
        field = crop_image(field, self.placement.get_place_recon_corners())
        theta = np.dot(self.placement.get_X_plane_recon_rectangle_pseudoinverse(), field.reshape(-1))
        ymin, ymax = self.placement.get_place_recon_corners()[0][0], self.placement.get_place_recon_corners()[0][1]
        plane = np.dot(self.placement.get_X_plane_recon_rectangle(), theta).reshape((ymax-ymin)%(cfg.image_size[0]+1), -1)
        return field-plane

class SpatialPhaseAveraging:
    def __init__(self,
                 position: Position,
                 placements: List[Placement],
                 timestep: int,
                 ):
        
        self.position: Position = position
        self.timestep: int = timestep
        self.placements: List[Placement] = placements
        self.holograms: List[Hologram] = self._generate_holograms()
        self.num_place: int = len(self.holograms)
        self.background: CplxImage = self._background()
        self.spatial_avg: CplxImage = self._spatial_avg()
    
    
    def _background(self) -> CplxImage:
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
        spatial_avg = self.holograms[0].get_cplx_image()
        spatial_avg /= self.background
        for i in range(1, self.num_place):
            cplx_image = self.holograms[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image, shift_vector = self._shift_image(spatial_avg, cplx_image)
            self.placements[i].set_shift_vector(shift_vector)
            cplx_image = self._subtract_phase_offset(cplx_image, spatial_avg, self.placements[i].get_place_image_roi())
            spatial_avg += cplx_image
        return spatial_avg/self.num_place
    
    def _generate_holograms(self) -> List[Hologram]:
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
    
    def _shift_image(self, reference_image: CplxImage, moving_image: CplxImage) -> (CplxImage, List[int]):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shift_vector, mode='constant')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shift_vector, mode='constant')
        return real+complex(0.,1.)*imaginary, shift_vector
        
    def _subtract_phase_offset(self, new: CplxImage, avg: CplxImage, mask: Mask) -> CplxImage:
        z= np.angle(np.multiply(new[mask==True],np.conj(avg[mask==True]))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new


class Pipeline:
    def __init__(
            self,
            base_dir: Union[str, Path],
            saving_dir: Union[str, Path] = None,
            restrict_positions: slice = None,
            restrict_timesteps: slice = None,
            ):
        
        self.base_dir: Path = Path(base_dir)
        self.saving_dir: Path = self._saving_dir(saving_dir)
        self.data_file_path: Path = None
        self.restrict_positions: slice = restrict_positions
        self.restrict_timesteps: range = restrict_timesteps
        self.positions: List[Position] = self._positions()
        self.timesteps: range = self._timesteps()
        self.image_settings_updated: bool = False
        self.image_count: int = 0
        
    def _get_mask_from_rectangle(self, image: Image) -> Mask:
        # Show the image and wait for user to select a rectangle
        cv2.imshow("Select a rectangle", image)
        rect = cv2.selectROI("Select a rectangle", image, False)
        cv2.destroyAllWindows()
        
        # Create a mask with the same shape as the image, initialized to zeros
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Set the pixels inside the rectangle to 1 in the mask
        x, y, w, h = rect
        mask[y:y+h, x:x+w] = 1
        
        return mask
        
    
    def _get_rectangle_coordinates(self, image: Image) -> List[List[int]]:
        # Show the image and wait for user to select a rectangle
        cv2.imshow("Select a rectangle", image)
        rect = cv2.selectROI("Select a rectangle", image, False)
        cv2.destroyAllWindows()
    
        # Extract coordinates of the rectangle
        x, y, w, h = rect
        ymin, ymax = y, y + h
        xmin, xmax = x, x + w
        
        return [[ymin, ymax], [xmin, xmax]]

    def _positions(self) -> List[Position]:
        all_positions =[Position(Path(f.path)) for f in os.scandir(self.base_dir) if f.is_dir()]
        if self.restrict_positions == None:
            return all_positions
        else:
            return all_positions[self.restrict_positions]
    
    def process(self):
        if cfg._LOADED is False:
            raise RuntimeError(
                "configuration has not been loaded, do so by executing sa.config.load_config"
            )
        for po in self.positions:
            placements = [Placement(place_dir=Path(str(f.path)), position=po) for f in os.scandir(str(po.pos_dir)) if f.is_dir()]
            last_phase_image = None
            for t in self.timesteps:
                start_image = time.time()
                spa = SpatialPhaseAveraging(po, placements, t)
                averaged_phase_image = get_result_unwrap(np.angle(spa.get_spatial_avg()))
                if last_phase_image is not None:
                    averaged_phase_image = self._temporal_shift_correction(last_phase_image, averaged_phase_image)
                phase_image =  averaged_phase_image[cfg.image_cut[0][0]:cfg.image_cut[0][1],cfg.image_cut[1][0]:cfg.image_cut[1][1]]
                
                if not self.image_settings_updated:
                    cfg.set_image_variables((cfg.KOALA_HOST.GetPhaseWidth(),cfg.KOALA_HOST.GetPhaseHeight()), cfg.KOALA_HOST.GetPxSizeUm()*1e-6, cfg.KOALA_HOST.GetLambdaNm(0)*1e-9)
                    self.data_file_path = self._write_data_file()
                    self.image_settings_updated = True
                
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
                    if cfg.DISPLAY_ALWAYS_ON:
                        shut_down_restart_koala()
                    else:
                        logout_login_koala()
                del spa
                
            del placements
            gc.collect()
            
    def _save_image(self, phase_image, save_pos_folder, t):
        if cfg.save_format == ".tif":
            ph = (((phase_image + np.pi/2) / np.pi) * 65535).astype(np.int16)
            fname = save_pos_folder + os.sep + f"pos{save_pos_folder[-5:]}cha1fra{str(t).zfill(5)}.tif"
            tifffile.imwrite(fname, ph)
            
        if cfg.save_format == ".bin":
            fname = save_pos_folder +"\\ph_timestep_" + str(t).zfill(5) + cfg.save_format
            binkoala.write_mat_bin(fname, phase_image, phase_image.shape[0], phase_image.shape[1], cfg.px_size, cfg.hconv, cfg.unit_code)
    
    def _saving_dir(self, saving_dir: Union[str, Path]) -> Path:
        if saving_dir == None:
            saving_dir = Path(str(self.base_dir) + " phase averages")
            if not os.path.exists(str(saving_dir)):
                os.makedirs(str(saving_dir))
        return Path(saving_dir)
    
    def select_positions_image_roi(self):
        for po in self.positions:
            p0_dir = Path(str(po.pos_dir) + os.sep + os.listdir(str(po.pos_dir))[0])
            p0 = Placement(place_dir=p0_dir, position=po)
            fname = Path(str(p0.place_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p0, focus = cfg.reconstruction_distance_guess)
            ph_image = np.angle(hologram.get_cplx_image())
            mask = self._get_mask_from_rectangle(ph_image)
            po.set_pos_image_roi(mask)
    
    def select_positions_recon_rectangle(self):
        for po in self.positions:
            p0_dir = Path(str(po.pos_dir) + os.sep + os.listdir(str(po.pos_dir))[0])
            p0 = Placement(place_dir=p0_dir, position=po)
            fname = Path(str(p0.place_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p0, focus = cfg.reconstruction_distance_guess)
            ph_image = np.angle(hologram.get_cplx_image())
            crop_coords = self._get_rectangle_coordinates(ph_image)
            po.set_pos_recon_corners(crop_coords)
            
    def _temporal_shift_correction(self, reference_image: Image, moving_image: Image) -> Image:
        shift_measured, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        return ndimage.shift(moving_image, shift=shift_vector, mode='wrap')
        
    def _timesteps(self) -> range:
        if self.restrict_timesteps == None:
            holo_path = str(self.base_dir)+os.sep+self.positions[0].pos_name + os.sep + "00001_00001\Holograms"
            first_holo = int(sorted(os.listdir(holo_path))[0][:5])
            num_timesteps = len(os.listdir(holo_path))
            all_timesteps = range(first_holo, first_holo + num_timesteps)
            return all_timesteps
        else:
            return self.restrict_timesteps
    
    def _update_data_file(self, spa, time):
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
        }
        
        with open(self.data_file_path, 'w') as file:
            json.dump(data, file, indent=4)
            

    def _write_data_file(self) -> Path:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        data_file_path = Path(self.saving_dir, f'data file {current_datetime}.json')
        
        data = {
            "settings": {
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
                "restart_koala": cfg.DISPLAY_ALWAYS_ON
            },
            "images": {}
        }
        
        with open(data_file_path, 'w') as file:
            json.dump(data, file, indent=4)  # Add indent parameter to make it readable
        
        return data_file_path
        