# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:24:03 2023

@author: SWW-Bc20
"""
import os
import math
from typing import Tuple, Optional
from pathlib import Path

BASE_PATH = os.path.normpath(os.path.dirname(__file__))
"Folder where this file is in"
_LOADED = False
"Checks if config is loaded"

koala_config_nr = None
"koala configuration number"

"repeating grid search, around previous minimum (recommended). If False a scipy.optimize.minimize is used"
nfevaluations : Tuple[int] = (10, 5, 5, 5)
"gridsize of nth repeating search"
focus_method: Tuple[str] = ("phase_sharpness", "std_amp", "std_amp", "std_amp") # 'std_amp', 'phase_sharpness', 'combined'
"functions used to find mimimum. phase_sharpness recommended to find general location of mimimum, std_amp or combined recommended to find exact minimum."
"If local_grid_search=False only the first function in focus_method tuple is used to find the minimum"
nfev_max: int = 100
"if minimum is found at the edges, an adjacent is used. If minimum is not found until nfev_max funciton evaluations iamge is labeled corrupted -> Message"


reconstruction_distance_low: float = -2.0
"Lowest tolerable focus distance. Minimization only searches above this distance. local_grid_search can find minium below, but deems the image corrupted"
reconstruction_distance_high: float = -1.4
"Highest tolerable focus distance. Minimization only searches below this distance. local_grid_search can find minium above, but deems the image corrupted"


plane_fit_order: int = 4
"""Images are not always flat. Some are only tilted others are on a rounded plane. This plane is calculated with the ordinary least squares method on the
plane that features (x and y pixel positions) are polynomialy extended. This is the order of expansion. Generally order 4 or 5 is recommended.
If the operator knows that the plane is less complex lower are fine aswell."""
use_amp: bool = True
"Is amplitude used for the spatial averaging of the image. True is recommended"

image_size: Tuple[int, int] = (800, 800)
"Input size of the Koala image. Is updated automatically"
px_size: float = 0.12976230680942535
"pixel size of the Koala image in micrometers. Is updated automatically"
hconv: float = 794/(2*math.pi)
"Conversion from degree into nanometers (optical path difference)"
unit_code: int = 1
"Unit code for koala. (0 -> no unit, 1 -> radians, 2 -> meters)"
image_cut: Tuple[Tuple[int]] = ((10, 710), (90, 790))
"Edges are not useable because of the spatial averaging. Image are cropped"
save_format: str = ".tif"
".tif, .bin or delta. If delta is selected images names are selected that work with delta. BF images are always saved as .tif"
save_as_bulk: bool = False

delta_assets_path: Path = Path(Path(BASE_PATH).parent, 'data', 'delta_assets')

additional_image_types = ('BF', 'PC')
"Is there a bf (brightfield) image and should it be evaluated"
bf_cut: Tuple[Tuple[int]] = ((512, 1536), (512, 1536))
"BF image is usually much bigger then the DHM phase image, only a part is used to speed up process"
bf_local_searches: int = 4
"number of local searches. Increases processing time and accuracy. Each search decreases search length by a factor of 3."
bf_rot_guess: float = 0.0
"Initial guess for rotation"
bf_rot_search_length: float = 2.0
"length of the rot search"
bf_zoom_guess: float = 0.905
"Initial guess for zoom"
bf_zoom_search_length: float = 0.2
"length of the zoom search"

"Is there a ph (brightfield) image and should it be evaluated"
pc_cut: Tuple[Tuple[int]] = ((412, 1436), (312, 1336))
"PC image is usually much bigger then the DHM phase image, only a part is used to speed up process"
pc_local_searches: int = 4
"number of local searches. Increases processing time and accuracy. Each search decreases search length by a factor of 3."
pc_rot_guess: float = 0.0
"Initial guess for rotation"
pc_rot_search_length: float = 2.0
"length of the rot search"
pc_zoom_guess: float = 0.905
"Initial guess for zoom"
pc_zoom_search_length: float = 0.2
"length of the zoom search"

delta_bacteria_core = "BF"
"Determines the image type which is used for bacteria segemntation."
model_file_bf_core_seg: Path = Path(delta_assets_path, 'models', 'unet_pads_bf_core_seg.hdf5')
"Unet for bf core segmenation"
model_file_ph_core_seg: Path = Path(delta_assets_path, 'models', 'unet_pads_ph_core_seg.hdf5')
"Unet for ph core segmentation"
model_file_bf_track: Path = Path(delta_assets_path, 'models', 'unet_pads_track.hdf5')
"Unet for bf tracking. Delta base tracking model works better, then tuned net -> base net"
model_file_ph_track: Path = Path(delta_assets_path, 'models', 'unet_pads_track.hdf5')
"Unet for ph tracking. Delta base tracking model works better, then tuned net -> base net"
model_file_ph_full_seg: Path = Path(delta_assets_path, 'models', 'unet_pads_full_ph_seg.hdf5')
"Full segmentaion of bacteria vs background. Used as mask for watershedding."

def load_config(koala_config_nrIn: int, nfevaluationsIn : Tuple[int] = None,
                focus_methodIn: Tuple[str] = None, reconstruction_distance_lowIn: float = None,
                reconstruction_distance_highIn: float = None, plane_fit_orderIn: int = None,
                use_ampIn: bool = None, save_formatIn: str = None, save_in_same_folderIn: bool = None):
    
    """
    This function laods the koala configuration and can change any configuration of the pipeline. If None are selected recommended caonfigurations are used.
    Configurations only reset to recommended configurations after restarting the python kernel.
    """
    
    # Update the configuration settings if arguments are provided
    global koala_config_nr
    koala_config_nr = koala_config_nrIn
    
    if nfevaluationsIn is not None:
        global nfevaluations
        nfevaluations = nfevaluationsIn

    if focus_methodIn is not None:
        global focus_method
        focus_method = focus_methodIn
    
    if reconstruction_distance_lowIn is not None:
        global reconstruction_distance_low
        reconstruction_distance_low = reconstruction_distance_lowIn
    
    if reconstruction_distance_highIn is not None:
        global reconstruction_distance_high
        reconstruction_distance_high = reconstruction_distance_highIn
    
    if plane_fit_orderIn is not None:
        global plane_fit_order
        plane_fit_order = plane_fit_orderIn
    
    if use_ampIn is not None:
        global use_amp
        use_amp = use_ampIn
        
    if save_formatIn is not None:
        global save_format
        save_format = save_formatIn
        
    if save_in_same_folderIn is not None:
        global save_in_same_folder
        save_in_same_folder = save_in_same_folderIn
    
    global _LOADED
    _LOADED = True

def set_image_variables(image_sizeIn: Tuple, px_sizeIn: float, laser_lambd: float):
    # hconv is hard coded and can not be changed, since Koala uses the wrong hconv
    global image_size
    image_size = image_sizeIn
    global px_size
    px_size = px_sizeIn
    global hconv
    hconv = laser_lambd/(2*math.pi)

def set_image_cut(image_cutIn: Tuple[Tuple[int]]):
    global image_cut
    image_cut = image_cutIn