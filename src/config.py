# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:24:03 2023

@author: SWW-Bc20
"""
import os
import math
from typing import Tuple
from pathlib import Path
import json

BASE_PATH = os.path.normpath(os.path.dirname(__file__))
"Folder where this file is in"
DELTA_ASSETS_PATH: Path = Path(Path(BASE_PATH).parent, 'data', 'delta_assets')

DEFAULT_CONFIG = {
    "koala_config_nr": None,
    # "repeating grid search, around previous minimum (recommended). If False a scipy.optimize.minimize is used"
    "nfevaluations": (10, 5, 5, 5),
    # "gridsize of nth repeating search"
    "focus_method": ("phase_sharpness", "std_amp", "std_amp", "std_amp"),
    # 'std_amp', 'phase_sharpness', 'combined'
    # "functions used to find mimimum. phase_sharpness recommended to find general location of mimimum,
    # std_amp or combined recommended to find exact minimum."
    # "If local_grid_search=False only the first function in focus_method tuple is used to find the minimum"
    "nfev_max": 100,
    # "if minimum is found at the edges, an adjacent is used.
    # If minimum is not found until nfev_max funciton evaluations iamge is labeled corrupted -> Message"
    "reconstruction_distance_low": -2.0,
    # "Lowest tolerable focus distance. Minimization only searches above this distance.
    # local_grid_search can find minium below, but deems the image corrupted"
    "reconstruction_distance_high": -1.4,
    # "Highest tolerable focus distance. Minimization only searches below this distance.
    # local_grid_search can find minium above, but deems the image corrupted"
    "plane_fit_order": 4,
    # """Images are not always flat. Some are only tilted others are on a rounded plane.
    # This plane is calculated with the ordinary least squares method on the
    # plane that features (x and y pixel positions) are polynomialy extended.
    # This is the order of expansion. Generally order 4 or 5 is recommended.
    # If the operator knows that the plane is less complex lower are fine aswell."""
    "use_amp": True,
    # "Is amplitude used for the spatial averaging of the image. True is recommended"
    "image_size": (800, 800),
    # "Input size of the Koala image. Is updated automatically"
    "px_size": 0.12976230680942535,
    # "pixel size of the Koala image in micrometers. Is updated automatically"
    "hconv": 794 / (2 * math.pi),
    # "Conversion from degree into nanometers (optical path difference)"
    "unit_code": 1,
    # "Unit code for koala. (0 -> no unit, 1 -> radians, 2 -> meters)"
    "image_cut": ((10, 710), (90, 790)),
    # "Edges are not useable because of the spatial averaging. Image are cropped"
    "save_format": ".tif",
    # ".tif, .bin or delta. If delta is selected images names are selected that work with delta.
    # BF images are always saved as .tif"
    "save_as_bulk": False,
    "additional_image_types": ('BF',),
    # "Is there a bf (brightfield) image and should it be evaluated"
    "bf_cut": ((512, 1536), (512, 1536)),
    # "BF image is usually much bigger then the DHM phase image, only a part is used to speed up process"
    "bf_local_searches": 4,
    # "number of local searches. Increases processing time and accuracy.
    # Each search decreases search length by a factor of 3."
    "bf_rot_guess": 0.0,
    # "Initial guess for rotation"
    "bf_rot_search_length": 2.0,
    # "length of the rot search"
    "bf_zoom_guess": 0.905,
    # "Initial guess for zoom"
    "bf_zoom_search_length": 0.2,
    # "length of the zoom search"
    # "Is there a ph (brightfield) image and should it be evaluated"
    "pc_cut": ((412, 1436), (312, 1336)),
    # "PC image is usually much bigger then the DHM phase image, only a part is used to speed up process"
    "pc_local_searches": 4,
    # "number of local searches. Increases processing time and accuracy.
    # Each search decreases search length by a factor of 3."
    "pc_rot_guess": 0.0,
    # "Initial guess for rotation"
    "pc_rot_search_length": 2.0,
    # "length of the rot search"
    "pc_zoom_guess": 0.905,
    # "Initial guess for zoom"
    "pc_zoom_search_length": 0.2,
    # "length of the zoom search"
    "delta_bacteria_core": "BF",
    # "Determines the image type which is used for bacteria segemntation."
    "model_file_bf_core_seg": str(Path(DELTA_ASSETS_PATH, 'models', 'unet_pads_bf_core_seg.hdf5')),
    # "Unet for bf core segmenation"
    "model_file_ph_core_seg": str(Path(DELTA_ASSETS_PATH, 'models', 'unet_pads_ph_core_seg.hdf5')),
    # "Unet for ph core segmentation"
    "model_file_bf_track": str(Path(DELTA_ASSETS_PATH, 'models', 'unet_pads_track.hdf5')),
    # "Unet for bf tracking. Delta base tracking model works better, then tuned net -> base net"
    "model_file_ph_track": str(Path(DELTA_ASSETS_PATH, 'models', 'unet_pads_track.hdf5')),
    # "Unet for ph tracking. Delta base tracking model works better, then tuned net -> base net"
    "model_file_ph_full_seg": str(Path(DELTA_ASSETS_PATH, 'models', 'unet_pads_full_ph_seg.hdf5')),
    # "Full segmentaion of bacteria vs background. Used as mask for watershedding."
}
    
class Config:
    def __init__(self, koala_config_nr = None, file_path=str(Path(BASE_PATH,'config.json'))):
        self.file_path = file_path
        self._config = None
        self.load_config()
        self.set_config_setting('koala_config_nr', koala_config_nr)

    def load_config(self):
        try:
            with open(self.file_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            print("Config file not found. Using default configuration.")
            self._config = DEFAULT_CONFIG

    def get_config_setting(self, setting_name):
        return self._config.get(setting_name, None)

    def set_config_setting(self, setting_name, value):
        self._config[setting_name] = value

    def save_config(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2)