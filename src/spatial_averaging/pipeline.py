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
from typing import cast, List, Union, Dict, Optional, Any, Tuple
from pathlib import Path
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, Bounds

from .utilities import cfg, get_result_unwrap, logout_login_koala, shut_down_restart_koala
from . import binkoala

Image = npt.NDArray[np.float32]

CplxImage = npt.NDArray[np.complex64]

Mask = npt.NDArray[np.uint8]

class Location:
    def __init__(
            self,
            loc_dir: Union[str, Path],
            ):
        
        self.loc_dir: Path = Path(loc_dir)
        self.loc_name: str = loc_dir.name
        self.loc_roi: Mask = np.ones(cfg.image_size, dtype=np.uint8)
        self.roi_selected: bool = False
        
    def get_loc_roi(self) -> Mask:
        return self.loc_roi
    
    def set_loc_roi(self, loc_roi: Mask):
        self.loc_roi = loc_roi.astype(np.uint8)
        self.roid_selected = True

class Position:
    def __init__(
            self,
            pos_dir: Union[str, Path],
            location: Location,
            ):
    
        self.pos_dir: Path = Path(pos_dir)
        self.location = location
        self.pos_name: str = pos_dir.name
        self.pos_roi: Mask = np.ones(cfg.image_size, dtype=np.uint8)
        self.shift_vector: Tuple = (0,0)
        self.x0_guess: float = cfg.reconstruction_distance_guess
        self.X_plane: npt.NDArray[np.float64] = None
        self.X_plane_roi: npt.NDArray[np.float64] = None
        self.X_plane_roi_pseudoinverse: npt.NDArray[np.float64] = None
    
    def get_pos_roi(self) -> Mask:
        return self.pos_roi
    
    def get_shift_vector(self) -> Tuple:
        return self.shift_vector
    
    def get_x0_guess(self) -> float:
        return self.x0_guess
    
    def get_X_plane(self) -> npt.NDArray[np.float64]:
        if self.X_plane is None:
            self._X_plane()
        return self.X_plane
    
    def get_X_plane_roi(self) -> npt.NDArray[np.float64]:
        if self.X_plane_roi is None:
            self._X_plane()
        return self.X_plane_roi
    
    def get_X_plane_roi_pseudoinverse(self) -> npt.NDArray[np.float64]:
        if self.X_plane_roi is None:
            self._X_plane()
        if self.X_plane_roi_pseudoinverse is None:
            self._X_plane_roi_pseudoinverse()
        return self.X_plane_roi_pseudoinverse
    
    def set_pos_roi(self):
        self.pos_roi = ndimage.shift(self.location.get_loc_roi(), shift=self.shift_vector, mode='wrap')
    
    def set_shift_vector(self, shift_vector: Tuple):
        self.shift_vector = shift_vector
        self.set_pos_roi()
        self._X_plane()
        self._X_plane_roi_pseudoinverse()
    
    def set_x0_guess(self, x0_guess: float):
        self.x0_guess = x0_guess
    
    def _X_plane(self):
        if cfg.plane_basis_vectors == "Polynomial":
            ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
            ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
            X1, X2 = np.mgrid[:cfg.image_size[0], :cfg.image_size[1]]
            X1_roi, X2_roi = X1[self.pos_roi==True], X2[self.pos_roi==True]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            X_roi = np.hstack((X1_roi.reshape(-1,1) , X2_roi.reshape(-1,1)))
            self.X_plane = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X)
            self.X_plane_roi = PolynomialFeatures(degree=cfg.plane_fit_order, include_bias=True).fit_transform(X_roi)
        else:
            print(cfg.plane_basis_vectors,  ' is not implemented')
    
    def _X_plane_roi_pseudoinverse(self):
        self.X_plane_roi_pseudoinverse = np.dot(np.linalg.inv(np.dot(self.X_plane_roi.transpose(), self.X_plane_roi)), self.X_plane_roi.transpose())
        
class Hologram:
    def __init__(self,
                 fname: Union[str, Path],
                 position: Position,
                 focus: float = None,
                 ):

        self.fname: Path = Path(fname)
        self.position: Position = position
        self.focus: float = focus # Focus distance
        self.focus_score: float = None # score of evaluatino function at the Focus point (minimum)
        self.cplx_image: Image = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self):
        cfg.KOALA_HOST.LoadHolo(str(self.fname),1)
        cfg.KOALA_HOST.SetUnwrap2DState(True)
        bounds = Bounds(lb=cfg.reconstrution_distance_low, ub=cfg.reconstrution_distance_high)
        res = minimize(self._evaluate_reconstruction_distance, [self.position.get_x0_guess()], method=cfg.optimizing_method, bounds=bounds)
        self.focus = res.x[0]
        self.position.set_x0_guess(self.focus)
        self.focus_score = res.fun
        self.cplx_image = self._cplx_image()
            
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
        return cplx_image
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance) -> float:
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
    
    def _evaluate_sobel_squared_std(self, gray_image) -> float:
        # Calculate gradient magnitude using Sobel filter
        grad_x = ndimage.sobel(gray_image, axis=0)
        grad_y = ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 2 + grad_y ** 2)
    
    def get_cplx_image(self) -> CplxImage:
        if self.cplx_image is None:
            if self.focus is None:
                self.calculate_focus()
            self.cplx_image = self._cplx_image()
        return self.cplx_image
    
    def _subtract_plane(self, field) -> CplxImage:
        theta = np.dot(self.position.get_X_plane_roi_pseudoinverse(), field[self.position.get_pos_roi()==True].reshape(-1))
        plane = np.dot(self.position.get_X_plane(), theta).reshape(field.shape[0], field.shape[1])
        return field-plane

class SpatialPhaseAveraging:
    def __init__(self,
                 location: Location,
                 positions: List[Position],
                 timestep: int,
                 ):
        
        self.location: Location = location
        self.timestep: int = timestep
        self.positions: List[Position] = positions
        self.num_pos: int = len(self.positions)
        self.holograms: List[Hologram] = self._generate_holograms()
        self.background: CplxImage = self._background()
        self.spatial_avg: CplxImage = self._spatial_avg()
    
    
    def _background(self) -> CplxImage:
        background = self.holograms[0].get_cplx_image()
        background = self._subtract_bacterias(background)
        for i in range(1, self.num_pos):
            cplx_image = self.holograms[i].get_cplx_image()
            cplx_image = self._subtract_bacterias(cplx_image)
            cplx_image = self._subtract_phase_offset(cplx_image, background, self.positions[i].get_pos_roi())
            background += cplx_image
        return background/self.num_pos
    
    def _spatial_avg(self) -> CplxImage:
        spatial_avg = self.holograms[0].get_cplx_image()
        spatial_avg /= self.background
        for i in range(1, self.num_pos):
            cplx_image = self.holograms[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image, shift_vector = self._shift_image(spatial_avg, cplx_image)
            self.positions[i].set_shift_vector(shift_vector)
            cplx_image = self._subtract_phase_offset(cplx_image, spatial_avg, self.positions[i].get_pos_roi())
            spatial_avg += cplx_image
        return spatial_avg/self.num_pos
    
    def _generate_holograms(self) -> List[Hologram]:
        holograms = []
        for p in self.positions:
            fname = Path(str(p.pos_dir) + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p)
            hologram.calculate_focus()
            # first guess is the focus point of the last image
            p.set_x0_guess(hologram.focus)
            holograms.append(hologram)
        return holograms
    
    def get_background(self) -> CplxImage:
        return self.background
    
    def get_spatial_avg(self) -> CplxImage:
        return self.spatial_avg
    
    def _shift_image(self, reference_image: CplxImage, moving_image: CplxImage) -> (CplxImage, Tuple):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shift_vector, mode='wrap')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shift_vector, mode='wrap')
        return real+complex(0.,1.)*imaginary, shift_vector
    
    def _subtract_bacterias(self, cplx_image: CplxImage) -> CplxImage:
        # subtracts pixel  that are far away from the mean and replaces them with the mean of the image
        # cut off value is determined by hand and has to be reestimated for different use cases
        ph = np.angle(cplx_image)
        ph[cfg.bacteria_cut_off<ph] = np.mean(ph[ph<cfg.bacteria_cut_off])
        return np.absolute(cplx_image)*np.exp(1j*ph)
        
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
            restrict_locations: slice = None,
            restrict_timesteps: slice = None,
            ):
        
        self.base_dir: Path = Path(base_dir)
        self.saving_dir: Path = self._saving_dir(saving_dir)
        self.restrict_locations: slice = restrict_locations
        self.restrict_timesteps: slice = restrict_timesteps
        self.locations: List[Location] = self._locations()
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
        
    def _locations(self) -> List[Location]:
        all_locations = [ Location(Path(f.path)) for f in os.scandir(self.base_dir) if f.is_dir()]
        if self.restrict_locations == None:
            return all_locations
        else:
            return all_locations[self.restrict_locations]
    
    def process(self):
        if cfg._LOADED is None:
            raise RuntimeError(
                "configuration has not been loaded, do so by executing sa.config.load_config"
            )
        for l in self.locations:
            positions = [Position(pos_dir=Path(str(f.path)), location=l) for f in os.scandir(str(l.loc_dir)) if f.is_dir()]
            last_phase_image = None
            for t in self.timesteps:
                start_image = time.time()
                spa = SpatialPhaseAveraging(l, positions, t)
                averaged_phase_image = get_result_unwrap(np.angle(spa.get_spatial_avg()))
                if last_phase_image is not None:
                    averaged_phase_image = self._temporal_shift_correction(last_phase_image, averaged_phase_image)
                
                if self.image_settings_updated:
                    cfg.set_image_variables((cfg.KOALA_HOST.GetPhaseWidth(),cfg.KOALA_HOST.GetPhaseHeight()), cfg.KOALA_HOST.GetPxSizeUm()*1e-6)
                    self.image_settings_updated = True
                
                save_loc_folder = str(self.saving_dir) + os.sep + l.loc_name
                if not os.path.exists(save_loc_folder):
                    os.makedirs(save_loc_folder)
                fname = save_loc_folder +"\\ph_timestep_"+str(t).zfill(5)+".bin"
                binkoala.write_mat_bin(fname, averaged_phase_image, cfg.image_size[0], cfg.image_size[1], cfg.px_size, cfg.hconv, cfg.unit_code)
                print(fname, "done in", np.round(time.time()-start_image,1), "seconds")
                
                self.image_count += 1
                if self.image_count % cfg.koala_reset_frequency == 0:
                    if cfg.DISPLAY_ALWAYS_ON:
                        shut_down_restart_koala()
                    else:
                        logout_login_koala()
                del spa
                
            del positions
            gc.collect()
    
    def _saving_dir(self, saving_dir: Union[str, Path]) -> Path:
        if saving_dir == None:
            saving_dir = Path(str(self.base_dir) + " phase averages")
            if not os.path.exists(str(saving_dir)):
                os.makedirs(str(saving_dir))
        return Path(saving_dir)
    
    def select_locations_roi(self):
        for l in self.locations:
            p0_dir = Path(str(l.loc_dir) + os.sep + os.listdir(str(l.loc_dir))[0])
            p0 = Position(pos_dir=p0_dir, location=l)
            fname = Path(str(p0.pos_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            hologram = Hologram(fname, p0, focus = cfg.reconstruction_distance_guess)
            ph_image = np.angle(hologram.get_cplx_image())
            mask = self._get_mask_from_rectangle(ph_image)
            l.set_loc_roi(mask)
            
    def _temporal_shift_correction(self, reference_image: Image, moving_image: Image) -> Image:
        shift_measured, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=10, normalization=None)
        shift_vector = (shift_measured[0],shift_measured[1])
        return ndimage.shift(moving_image, shift=shift_vector, mode='wrap')
        
    def _timesteps(self) -> range:
        all_timesteps = range(len(os.listdir(str(self.base_dir)+os.sep+self.locations[0].loc_name + os.sep + "00001_00001\Holograms")))
        if self.restrict_timesteps == None:
            return all_timesteps
        else:
            return all_timesteps[self.restrict_timesteps]