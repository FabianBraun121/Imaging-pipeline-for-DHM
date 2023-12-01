# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:00:59 2023

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

sys.path.append("..")
from src.config import Config
from src.spatial_averaging.utilities import (
    gradient_squared, grid_search_2d, zoom,
    PolynomialPlaneSubtractor, Koala, ValueTracker
)

from . import binkoala

Image = npt.NDArray[np.float32]
CplxImage = npt.NDArray[np.complex64]
Mask = npt.NDArray[np.uint8]
Matrix = np.ndarray


class PolynomialPlaneSubtractorReconImage(PolynomialPlaneSubtractor):
    pass


class PolynomialPlaneSubtractorFullImage(PolynomialPlaneSubtractor):
    pass


class RotationZoomTracker(ValueTracker):
    def calculate_average(self):
        rotation_median = np.median(np.array(self.value_list)[:, 0])
        zoom_median = np.median(np.array(self.value_list)[:, 1])
        self.value = (rotation_median, zoom_median)


class ImageTracker(ValueTracker):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._check_existence()
    
    def calculate_average(self):
        self.value = np.mean(np.abs(self.value_list), axis=0) * np.exp(1j * np.mean(np.angle(self.value_list), axis=0)).astype(np.complex64)
        tifffile.imwrite(self.filename, self.value)
        
    def _check_existence(self):
        if os.path.isfile(self.filename):
            self.value = np.array(tifffile.imread(self.filename))


class Position:
    def __init__(
            self,
            pos_dir: Union[str, Path],
            ):
        self.pos_dir: Path = Path(pos_dir)
        "Path to position"
        self.pos_name: str = pos_dir.name
        "position name, e.g. 00001"
        self.recon_corners = None
        "Reconstruction distance is only determined in this recangle"
        self.trackers: dict = dict()
        "Recurring features like background ar rotation/zoom can be safed here"
    
    def append_tracker_value(self, tracker_name, value):
        assert tracker_name in self.trackers, "tracker not initialised yet"
        self.trackers[tracker_name].append_value(value)
        
    def get_tracker_value(self, tracker_name, filename=None):
        if tracker_name in self.trackers:
            return self.trackers[tracker_name].value
        else:
            self.initialise_tracker(tracker_name, filename=filename)
            return self.trackers[tracker_name].value
    
    def initialise_tracker(self, tracker_name, filename=None):
        if filename is None:
            self.trackers[tracker_name] = RotationZoomTracker()
        else:
            self.trackers[tracker_name] = ImageTracker(filename)

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
        self.shift_vector: Tuple[int] = (0,0)
        "shif relative to the first placement, calculated with phase cross correlation"
    
        
class DHMImage:
    """
        Class handles the reconstruction of the dhm Image. 
        It interacts with Koala, can find the true focus distance and saves the complex image at the focus distance.
    """
    def __init__(self,
                 config: Config,
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
            Class with informations about the placement of the dhm Image.
        focus : float, optional
            Focus distance, can be found with self.calculate_focus. The default is None.

        Returns
        -------
        None.

        """
        
        self.cfg: dict = config
        self.fname: Path = Path(fname)
        "Path to the holograms .tif file"
        self.corrupted = self._check_corrupted()
        "If the image is corruped in any way, this is turned to true. This hologram is not considered in the further workflow anymore."
        self.placement: Placement = placement
        "Class with informations about the placement"
        self.focus: float = focus
        "Focus distance, can be found with self.calculate_focus. The default is None."
        self.nfev: int = 0
        "number of function evaluations. Very high number of evaluations indicate hard cost function, which indicate corrupted image"
        self.cplx_image: CplxImage = None # as soon as the focus point is found this function is evaluated
        "complex image at the focus distance, is added when calculating the focus."
    
    def calculate_focus(self):
        Koala.load_hologram(self.fname)
        "starting search grid"
        xmin, xmax = self.cfg.get_config_setting('reconstruction_distance_low'), self.cfg.get_config_setting('reconstruction_distance_high')
        nfev_max = self.cfg.get_config_setting('nfev_max')
        for i in range(len(self.cfg.get_config_setting('nfevaluations'))):
            nfevaluations = self.cfg.get_config_setting('nfevaluations')[i]
            focus_method = self.cfg.get_config_setting('focus_method')[i]
            x = np.linspace(xmin, xmax, nfevaluations)
            y = self._evatluate_reconstruction_distances(x, focus_method)
            while np.argmin(y) == 0 and self.nfev<nfev_max: # if first element is the minimum, appending grid is tested
                x = np.linspace(xmin-(xmax-xmin), xmin, nfevaluations)
                xmin, xmax = x[0], x[-1]
                y = self._evatluate_reconstruction_distances(x, focus_method)
            while np.argmin(y) == len(x)-1 and self.nfev<nfev_max: # if last element is the minimum, appending grid is tested
                x = np.linspace(xmax, xmax+(xmax-xmin), nfevaluations)
                xmin, xmax = x[0], x[-1]
                y = self._evatluate_reconstruction_distances(x, focus_method)
            if nfev_max<self.nfev:
                print(f'{self.fname} could not find a focus point')
                self.corrupted = True
                break
            
            "adjusting search grid"
            spacing = x[1] - x[0]
            xmin = x[np.argmin(y)] - spacing/2
            xmax = x[np.argmin(y)] + spacing/2
            
        self.focus = x[np.argmin(y)]
        if self.focus<self.cfg.get_config_setting('reconstruction_distance_low') or self.cfg.get_config_setting('reconstruction_distance_high')<self.focus:
            print(f'{self.fname} focus is out of borders with {np.round(self.focus,3)}')
            self.corrupted = True
        self._cplx_image()
    
    def _evatluate_reconstruction_distances(self, x, method):
        y = np.zeros_like(x)
        self.nfev += len(x)
        for i in range(len(x)):
            Koala.set_reconstruction_distance(x[i])
            if method == "phase_sharpness":
                y[i] = self._evaluate_phase_sharpness()
            elif method == "std_amp":
                y[i] = self._evaluate_std_amp()
            elif method == "combined":
                y[i] = self._evaluate_combined()
            else:
                raise RuntimeError(f'Evaluation function {method} not implemented')
        return y
    
    def _evaluate_phase_sharpness(self):
        ph = Koala.get_phase_image()
        if self.placement.position.recon_corners is not None:
            recon_corners = self.placement.position.recon_corners
            ph = ph[recon_corners[0][0]:recon_corners[0][1], recon_corners[1][0]:recon_corners[1][1]]
        ph = PolynomialPlaneSubtractorReconImage.subtract_plane(ph, self.cfg.get_config_setting('plane_fit_order'))
        ph = ph.clip(min=0)
        # Calculate gradient magnitude using Sobel filter
        grad_x = ndimage.sobel(ph, axis=0)
        grad_y = ndimage.sobel(ph, axis=1)
        # Calculate std squared sobel sharpness score
        return -np.std(grad_x ** 4 + grad_y ** 4)
    
    def _evaluate_std_amp(self):
        amp = Koala.get_intensity_image()
        if self.placement.position.recon_corners is not None:
            recon_corners = self.placement.position.recon_corners
            amp = amp[recon_corners[0][0]:recon_corners[0][1], recon_corners[1][0]:recon_corners[1][1]]
        amp = PolynomialPlaneSubtractorReconImage.subtract_plane(amp, self.cfg.get_config_setting('plane_fit_order'))
        return np.std(amp)
    
    def _evaluate_combined(self):
        amp = Koala.get_intensity_image()
        ph = Koala.get_phase_image()
        if self.placement.position.recon_corners is not None:
            recon_corners = self.placement.position.recon_corners
            amp = amp[recon_corners[0][0]:recon_corners[0][1], recon_corners[1][0]:recon_corners[1][1]]
            ph = ph[recon_corners[0][0]:recon_corners[0][1], recon_corners[1][0]:recon_corners[1][1]]
        amp = PolynomialPlaneSubtractorReconImage.subtract_plane(amp, self.cfg.get_config_setting('plane_fit_order'))
        ph = PolynomialPlaneSubtractorReconImage.subtract_plane(ph, self.cfg.get_config_setting('plane_fit_order'))
        return -np.std(ph)/np.std(amp)
        
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
        Koala.load_hologram(self.fname)
        Koala.set_reconstruction_distance(self.focus)
        amp = Koala.get_intensity_image()
        ph = Koala.get_phase_image()
        ph = PolynomialPlaneSubtractorFullImage.subtract_plane(ph, self.cfg.get_config_setting('plane_fit_order'))
        if self.cfg.get_config_setting('use_amp'):
            amp = Koala.get_intensity_image()
            self.cplx_image = amp*np.exp(complex(0.,1.)*ph)
        else:
            self.cplx_image = np.exp(complex(0.,1.)*ph)
        
    def get_cplx_image(self) -> CplxImage:
        return self.cplx_image


class SpatialPhaseAveraging:
    """
        Class subtracts the background and spatially averages a set of overlapping images.
    """
    def __init__(self,
                 config: Config,
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
        self.cfg = config
        "Reference to the config instance"
        self.position: Position = position
        "Position class with all the relevant informations about the position"
        self.timestep: int = timestep
        "timestep of the images which are averaged"
        self.corrupted: bool = False
        "If all holograms in the spatial averaging are corrupted, spa is also corrupted"
        self.placements: List[Placement] = placements
        "list of all placements. Placement is a class with informations about the placement of the hologram"
        self.dhm_images: List[DHMImage] = self._generate_dhm_images()
        "list with the hologrmas class for each of the placements at this timestep."
        self.num_place: int = len(self.dhm_images)
        "number of holograms, which are in use (corruped holograms are not in use)."
        self.background: CplxImage = self._background()
        "background of the images"
        self.spatial_avg: CplxImage = self._spatial_avg()
        "spatial average of all the holograms"
    
    
    def _background(self) -> CplxImage:
        if self.corrupted:
            return None
        "if position has a background this is taken. Else the background of this timestep is calculated. Appends this background to the position."
        if self.position.get_tracker_value('background', str(self.position.pos_dir)+os.sep+'background.tif') is not None:
            return self.position.get_tracker_value('background', str(self.position.pos_dir)+os.sep+'background.tif')
        else:
            image_size = self.cfg.get_config_setting('image_size')
            images = np.zeros((len(self.dhm_images), image_size[0], image_size[1]), dtype=np.complex64)
            for i in range(len(self.dhm_images)):
                images[i] = self.dhm_images[i].get_cplx_image()
            background = (np.median(np.abs(images), axis=0)*np.exp(1j*np.median(np.angle(images), axis=0))).astype(np.complex64)
            
            self.position.append_tracker_value('background', background)
            return background
    
    def _spatial_avg(self) -> CplxImage:
        if self.corrupted:
            return None
        "spatial average of all the images. (repeat for each image(Image -> background subtracted ->shifted->added))/number of images"
        spatial_avg = self.dhm_images[0].get_cplx_image()
        spatial_avg /= self.background
        for i in range(1, self.num_place):
            cplx_image = self.dhm_images[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image, shift_vector = self._shift_image(spatial_avg, cplx_image)
            self.placements[i].shift_vector = shift_vector
            cplx_image = self._subtract_phase_offset(spatial_avg, cplx_image)
            spatial_avg += cplx_image
        return spatial_avg/self.num_place
    
    def _generate_dhm_images(self) -> List[DHMImage]:
        "generates the holograms of all placements and calculates their focus. Corruped holograms are rejected."
        dhm_images = []
        for pl in self.placements:
            fname = Path(str(pl.place_dir) + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif")
            dhm_image = DHMImage(self.cfg, fname, pl)
            if dhm_image.corrupted:
                continue
            dhm_image.calculate_focus()
            dhm_images.append(dhm_image)
        dhm_images = [i for i in dhm_images if not i.corrupted]
        if len(dhm_images)==0:
            self.corrupted = True
        return dhm_images
    
    def get_background(self) -> CplxImage:
        return self.background.copy()
    
    def get_spatial_avg(self) -> CplxImage:
        return self.spatial_avg.copy()
    
    def _shift_image(self, reference_image: CplxImage, moving_image: CplxImage) -> (CplxImage, Tuple[int]):
        "calculates the shift between tow images."
        # increase in importance to the higher areas (bacteria)
        ref = np.exp(3*np.angle(reference_image))
        mov = np.exp(3*np.angle(moving_image))
        try: # from scikit-image version 0.19.1 they added normalization. base configuration is 'phase', but None works better
            shift_measured, _, __ = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None, return_error='always')
        except TypeError: # Invalid argument normalization
            shift_measured, _, __ = phase_cross_correlation(ref, mov, upsample_factor=10, return_error=True)
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=tuple(shift_measured), mode='constant')
        imaginary = ndimage.shift(np.imag(moving_image), shift=tuple(shift_measured), mode='constant')
        shift_vector = (int(np.round(shift_measured[0],0)), int(np.round(shift_measured[1],0)))
        return real+complex(0.,1.)*imaginary, shift_vector
        
    def _subtract_phase_offset(self, avg: CplxImage, new: CplxImage) -> CplxImage:
        "aligns the phases of the different iamges."
        z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
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
            config: Config,
            base_dir: Union[str, Path] = None, 
            saving_dir: Union[str, Path] = None,
            restrict_positions: Tuple[int, int] = None,
            restrict_timesteps: Tuple[int, int] = None,
            ):
        """
        Parameters
        ----------
        base_dir : Union[str, Path]
            Path to the directory where the experiment is saved.
        saving_dir : Union[str, Path], optional
            Path to the directory where the processed images should be saved. The default is None.
        restrict_positions : tuple, optional
            Slice of the positions that should be processed. The default is None.
        restrict_timesteps : range, optional
            Range of the timesteps that should be processed. The default is None.
            
        Returns
        -------
        None.

        """
        self.cfg: Config = config
        "All settings are saved in this config class"
        if base_dir is not None:
            self.cfg.set_config_setting('base_dir', str(base_dir))
        "Path to the directory where the experiment is saved"
        self.cfg.set_config_setting('saving_dir', str(self._saving_dir(saving_dir)))
        if not os.path.exists(self.cfg.get_config_setting('saving_dir')):
            os.makedirs(self.cfg.get_config_setting('saving_dir'))
        "Path to the directory where the processed images should be saved"
        if restrict_positions is not None:
            self.cfg.set_config_setting('restrict_positions', restrict_positions)
        "Slice of the positions that should be processed"
        if restrict_timesteps is not None:
            self.cfg.set_config_setting('restrict_timesteps', restrict_timesteps)
        "Range of the timesteps that should be processed"
        self.cfg.save_config(str(Path(self.cfg.get_config_setting('saving_dir'),'config.json')))
        "Saves the configuration file to the output folder"
        self.positions: List[Position] = self._positions()
        "List of the positions that are processed"
        self.first_timestep: int = None
        "number of the first timestep in the time-lapse (normally 0 or 1)"
        self.timesteps: range = self._timesteps()
        "Range of the timesteps that are processed"
        self.image_settings_updated: bool = False
        "Checks if the image settings in the config files are updated"
        self.data_file_path: Path = None
        "Path where the information about the processing is saved"
        
        Koala.connect(self.cfg.get_config_setting('koala_config_nr'))
        if self.cfg.get_config_setting('recon_rectangle'):
            self.select_positions_recon_rectangle(same_for_all_pos = self.cfg.get_config_setting('recon_all_the_same_var'),
                                                  recon_corners= self.cfg.get_config_setting('recon_corners'))
            
    
    def _get_last_phase_image(self, po: Position, save_ph_pos: str, t: int):
        save_format = self.cfg.get_config_setting('save_format')
        if self.cfg.get_config_setting('save_as_bulk'):
            fname = f'{save_ph_pos}//pos_{po.pos_name}_timestep_{str(t).zfill(5)}_PH{save_format}'
        else:
            fname = f'{save_ph_pos}//{str(t).zfill(5)}_PH{save_format}'
        if os.path.isfile(fname):
            if save_format == '.tif':
                return tifffile.imread(fname)
            elif save_format == '.bin':
                return binkoala.read_mat_bin(fname)[0]
            else:
                raise RuntimeError(f'Unknown save format: {save_format}')
        else:
            return None
    
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
        all_positions =[Position(Path(f.path)) for f in os.scandir(self.cfg.get_config_setting('base_dir')) if f.is_dir()]
        if self.cfg.get_config_setting('restrict_positions') == None:
            return all_positions
        else:
            min_pos, max_pos = self.cfg.get_config_setting('restrict_positions')
            return all_positions[min_pos:max_pos]
        
    def process(self):
        for po in self.positions:
            placements = [Placement(place_dir=Path(str(f.path)), position=po) for f in os.scandir(str(po.pos_dir)) if f.is_dir()] # list of all placements
            for t in self.timesteps:
                start_image = time.time()
                spa = SpatialPhaseAveraging(self.cfg, po, placements, t)
                if spa.corrupted:    
                    continue
                phase_image = np.angle(spa.get_spatial_avg()).astype(np.float32)
                
                if not self.image_settings_updated:
                    self._set_image_variables()
                    self._update_image_cut(spa)
                    self.data_file_path = self._write_data_file()
                    self.image_settings_updated = True
                
                if self.cfg.get_config_setting('save_as_bulk'):
                    save_ph_pos = str(self.cfg.get_config_setting('saving_dir')) + os.sep + 'PH'
                else:
                    save_ph_pos = str(self.cfg.get_config_setting('saving_dir')) + os.sep + po.pos_name
                if not os.path.exists(save_ph_pos):
                    os.makedirs(save_ph_pos)
                
                last_phase_image = self._get_last_phase_image(po, save_ph_pos, t)
                phase_image = self._temporal_shift_correction(phase_image, last_phase_image)
                
                for image_type in self.cfg.get_config_setting('additional_image_types'):
                    self._save_aligned_image_of_various_types(phase_image, po, t, image_type)
                
                self._save_image(phase_image, po, save_ph_pos, t)
                
                duration_timestep = np.round(time.time()-start_image,1)
                print(f"pos: {po.pos_name}, timestep: {t} done in {duration_timestep} seconds")
                self._update_data_file(spa, duration_timestep)
                
                del spa
                
            del placements
            gc.collect()
    
    def _save_aligned_image_of_various_types(self, phase_image: Image, po: Position, t: int, image_type: str) -> Image:
        it = image_type.lower() # since this is used ofthen
        image_cut = self.cfg.get_config_setting(f'{it}_cut')
        fname = Path(po.pos_dir, f'{str(t).zfill(5)}_{image_type}.tif')
        if not os.path.isfile(fname):
            raise RuntimeWarning(f'{fname} does not exist')
            return
        image = tifffile.imread(fname)
        image = np.fliplr(image)
        image = trans.rotate(image, -90, mode="edge")[image_cut[0][0]:image_cut[0][1], image_cut[1][0]:image_cut[1][1]]
        ph_ = np.zeros(image.shape)
        ph_[:phase_image.shape[0], :phase_image.shape[1]] = phase_image
        phase_ = gradient_squared(ph_)
        image_ = gradient_squared(image)
        if po.get_tracker_value(image_type) is None:
            rot, zoomlevel = grid_search_2d(phase_, image_, self.cfg.get_config_setting(f'{it}_rot_guess'),
                                            self.cfg.get_config_setting(f'{it}_zoom_guess'), self.cfg.get_config_setting(f'{it}_rot_search_length'),
                                            self.cfg.get_config_setting(f'{it}_zoom_search_length'), self.cfg.get_config_setting(f'{it}_local_searches'))
            po.append_tracker_value(image_type, (rot, zoomlevel))
        else:
            rot, zoomlevel = po.get_tracker_value(image_type)
        image_rz = zoom(trans.rotate(image_, rot, mode="edge"),zoomlevel)
        try:
            shift_measured = phase_cross_correlation(phase_, image_rz, upsample_factor=10, normalization=None)[0]
        except:
            shift_measured = phase_cross_correlation(phase_, image_rz, upsample_factor=10)[0]
        image = ndimage.shift(zoom(trans.rotate(image, rot, mode="edge"),zoomlevel), tuple(shift_measured))
        image = image[:phase_image.shape[0], :phase_image.shape[1]]
        
        if self.cfg.get_config_setting('save_as_bulk'):
            save_folder = f'{str(self.cfg.get_config_setting("saving_dir"))}//{image_type}'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_fname = f'{save_folder}//pos_{po.pos_name}_timestep_{str(t).zfill(5)}_{image_type}.tif'
        else:
            save_fname = f'{str(self.cfg.get_config_setting("saving_dir"))}//{po.pos_name}//{str(t).zfill(5)}_{image_type}.tif'
        tifffile.imwrite(save_fname, image)
    
    def _save_image(self, phase_image: Image, po: Position, save_ph_pos: str, t: int):
        "saves the images in the selected format. Either .tif or .bin"
        save_format = self.cfg.get_config_setting('save_format')
        if self.cfg.get_config_setting('save_as_bulk'):
            fname = f'{save_ph_pos}//pos_{po.pos_name}_timestep_{str(t).zfill(5)}_PH{save_format}'
        else:
            fname = f'{save_ph_pos}//{str(t).zfill(5)}_PH{save_format}'
            
        if save_format == ".tif":
            tifffile.imwrite(fname, phase_image)
        elif save_format == ".bin":
            binkoala.write_mat_bin(fname, phase_image, phase_image.shape[0], phase_image.shape[1], self.cfg.get_config_setting('px_size'),
                                   self.cfg.get_config_setting('hconv'), self.cfg.get_config_setting('unit_code'))
        else:
            raise RuntimeError(f'Unknown saving format: {save_format}')
    
    def _saving_dir(self, saving_dir: Union[str, Path]) -> Path:
        "returns the saveing dir"
        if saving_dir == None:
            if self.cfg.get_config_setting('saving_dir') == None:
                saving_dir = Path(str(self.cfg.get_config_setting('base_dir')) + " processed")
            else:
                return self.cfg.get_config_setting('saving_dir')
        return Path(saving_dir)
    
    def select_positions_recon_rectangle(self, same_for_all_pos: bool = False, recon_corners: Tuple[Tuple[int]] = None):
        "allows for selection of the reconstruction rectangle (part where the focusing mehtod is applied), for each position."
        for po in self.positions:
            if recon_corners is not None:
                po.recon_corners = recon_corners
                continue
            p0_dir = Path(str(po.pos_dir) + os.sep + [d for d in os.listdir(str(po.pos_dir)) if os.path.isdir(Path(po.pos_dir,d))][0])
            p0 = Placement(place_dir=p0_dir, position=po)
            fname = Path(str(p0.place_dir) + os.sep + "Holograms" + os.sep + str(self.timesteps[0]).zfill(5) + "_holo.tif")
            dhm_image = DHMImage(fname, p0, focus = (self.cfg.get_config_setting('reconstruction_distance_low')+self.cfg.get_config_setting('reconstruction_distance_high'))/2)
            ph_image = np.angle(dhm_image.get_cplx_image())
            recon_corners = self._get_rectangle_coordinates(ph_image)
            po.recon_corners = recon_corners
    
    def _set_image_variables(self):
        self.cfg.set_config_setting('image_size', (Koala._host.GetPhaseWidth(),Koala._host.GetPhaseHeight()))
        self.cfg.set_config_setting('px_size', Koala._host.GetPxSizeUm())
        self.cfg.set_config_setting('hconv', Koala._host.GetLambdaNm(0))
        
    def _temporal_shift_correction(self, current_image: Image, last_image: Image) -> Image:
        "images can move overtime, this function corrects for the shift. Movement is due to different focus distances"
        if last_image is None:
            image_cut = self.cfg.get_config_setting('image_cut')
            return current_image[image_cut[0][0]:image_cut[0][1],image_cut[1][0]:image_cut[1][1]]
        else:
            last_image_ = np.zeros_like(current_image)
            last_image_[:last_image.shape[0], :last_image.shape[1]] = last_image
            
            try: # from scikit-image version 0.19.1 they added normalization. base configuration is 'phase', but None works better
                shift_measured, _, __ = phase_cross_correlation(last_image_, current_image, upsample_factor=10, normalization=None, return_error='always')
            except TypeError: # Invalid argument normalization
                shift_measured, _, __ = phase_cross_correlation(last_image_, current_image, upsample_factor=10, return_error=True)
            current_image = ndimage.shift(current_image, shift=tuple(shift_measured), mode='constant')
            return current_image[:last_image.shape[0], :last_image.shape[1]]
        
    def _timesteps(self) -> range:
        "returns the range of the timesteps processed"
        holo_path = str(self.cfg.get_config_setting('base_dir'))+os.sep+self.positions[0].pos_name + os.sep + "00001_00001\Holograms"
        self.first_timestep = int(sorted(os.listdir(holo_path))[0][:5])
        num_timesteps = len(os.listdir(holo_path))
        all_timesteps = range(self.first_timestep, self.first_timestep + num_timesteps)
        if self.cfg.get_config_setting('restrict_timesteps') == None:
            return all_timesteps
        else:
            min_timestep, max_timestep = self.cfg.get_config_setting('restrict_timesteps')
            return all_timesteps[min_timestep:max_timestep]
    
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
            "foci": tuple(dhm_image.focus for dhm_image in spa.dhm_images),
            "function_evaluations": int(np.sum([dhm_image.nfev for dhm_image in spa.dhm_images])),
            "shift_x": tuple(int(placement.shift_vector[1]) for placement in spa.placements),
            "shift_y": tuple(int(placement.shift_vector[0]) for placement in spa.placements),
        }
        
        for image_type in self.cfg.get_config_setting('additional_image_types'):
            if spa.position.trackers[image_type].value is not None:
                rot, zoom = spa.position.trackers[image_type].value
            else:
                rot, zoom = spa.position.trackers[image_type].value_list[-1]
            data["images"][image_name][f"{image_type.lower()}_rotation"] = rot
            data["images"][image_name][f"{image_type.lower()}_zoom"] = zoom
        
        with open(self.data_file_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    def _update_image_cut(self, spa: SpatialPhaseAveraging):
        "shifts the image cut to account for temporal movement"
        y_shifts = np.array([placement.shift_vector[0] for placement in spa.placements])
        x_shifts = np.array([placement.shift_vector[1] for placement in spa.placements])
        y_midpoint = np.round(np.mean([np.min(y_shifts), np.max(y_shifts)]), 0)
        x_midpoint = np.round(np.mean([np.min(x_shifts), np.max(x_shifts)]), 0)
        image_size = self.cfg.get_config_setting('image_size')
        image_cut = ((50+int(y_midpoint), image_size[0]-50+int(y_midpoint)), (50+int(x_midpoint), image_size[1]-50+int(x_midpoint)))
        self.cfg.set_config_setting('image_cut', image_cut)

    def _write_data_file(self) -> Path:
        "writes an data file with the informations about the processing"
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        data_file_path = Path(self.cfg.get_config_setting('saving_dir'), f'data file {current_datetime}.json')
        
        data = {
            "settings": {
                "base_dir": str(self.cfg.get_config_setting('base_dir')),
                "koala_configuration": Koala._config_number,
                "nfevaluations": self.cfg.get_config_setting('nfevaluations'),
                "final_grid_spacing": (self.cfg.get_config_setting('reconstruction_distance_high')
                                       -self.cfg.get_config_setting('reconstruction_distance_low')
                                       )/np.prod([f-1 for f in self.cfg.get_config_setting('nfevaluations')]),
                "nfev_max": self.cfg.get_config_setting('nfev_max'),
                "reconstruction_distance_low": self.cfg.get_config_setting('reconstruction_distance_low'),
                "reconstruction_distance_high": self.cfg.get_config_setting('reconstruction_distance_high'),
                "plane_fit_order": self.cfg.get_config_setting('plane_fit_order'),
                "use_amp": self.cfg.get_config_setting('use_amp'),
                "image_size": self.cfg.get_config_setting('image_size'),
                "px_size": self.cfg.get_config_setting('px_size'),
                "hconv": self.cfg.get_config_setting('hconv'),
                "unit_code": self.cfg.get_config_setting('unit_code'),
                "image_cut": self.cfg.get_config_setting('image_cut'),
                "save_format": self.cfg.get_config_setting('save_format'),
            },
            "images": {}
        }
        with open(data_file_path, 'w') as file:
            json.dump(data, file, indent=4)  # Add indent parameter to make it readable
        return data_file_path
        