# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:24:03 2023

@author: SWW-Bc20
"""
import math
from typing import Tuple, Optional

_LOADED = False
"Checks if config is loaded"
KOALA_HOST = None
"koala host, remote connection with koala"


koala_config_nr = None
"koala configuration number"
display_always_on: bool = True
"restarting koala needs a grahical interface. Connection to display needs to be on. Display itself can be off"

local_grid_search: bool = True
"repeating grid search, around previous minimum (recommended). If False a scipy.optimize.minimize is used"
nfevaluations : Tuple[int] = (10, 5, 5, 5, 5)
"gridsize of nth repeating search"
focus_method: Tuple[str] = ("std_ph_sobel", "std_amp", "std_amp", "std_amp", "std_amp") # 'std_amp', 'sobel_squared_std', 'combined'
"functions used to find mimimum. std_ph_sobel recommended to find general location of mimimum, std_amp or combined recommended to find exact minimum."
"If local_grid_search=False only the first function in focus_method tuple is used to find the minimum"
nfev_max: int = 200
"if minimum is found at the edges, an adjacent is used. If minimum is not found until nfev_max funciton evaluations iamge is labeled corrupted -> Message"

optimizing_method: str = "Powell" # 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA','SLSQP', 'trust-constr'
"If local_grid_search=False scipy.optimize.minimize is used. Method: Powell was found to work best"
tolerance: Optional[float] = None
"Tolerance for termination of minimization. None is recommended"

reconstruction_distance_low: float = -3.0
"Lowest tolerable focus distance. Minimization only searches above this distance. local_grid_search can find minium below, but deems the image corrupted"
reconstruction_distance_high: float = -1.0
"Highest tolerable focus distance. Minimization only searches below this distance. local_grid_search can find minium above, but deems the image corrupted"
reconstruction_distance_guess: float = -2.3
"This is the best guess of the reconstruction distance of the operator."

plane_fit_order: int = 4
"""Images are not always flat. Some are only tilted others are on a rounded plane. This plane is calculated with the ordinary least squares method on the
plane that features (x and y pixel positions) are polynomialy extended. This is the order of expansion. Generally order 4 or 5 is recommended.
If the operator knows that the plane is less complex lower are fine aswell."""
use_amp: bool = True
"Is amplitude used for the spatial averaging of the image. True is recommended"

image_size: Tuple[int, int] = (800, 800)
"Input size of the Koala image. Is updated automatically"
px_size: float = 0.12976230680942535*1e-6
"pixel size of the Koala image in meters. Is updated automatically"
hconv: float = 794*1e-9/(2*math.pi)
"Conversion from degree into meters (optical path difference)"
unit_code : int = 1
"Unit code for koala. (0 -> no unit, 1 -> radians, 2 -> meters)"
image_cut : Tuple[Tuple[int]] = ((10, 710), (90, 790))
"Edges are not useable because of the spatial averaging. Image are cropped"
save_format: str = ".tif"
".tif or .bin. If image is also sent through delta it has to be .tif"

koala_reset_frequency: int = 20
"Koala slows down with time (due to accumulation of memory). Periodic restart is required. If local_grid_search=True frequency 20 is recommended, if False 10."

dilute_cells: int = 2
"""Some cells are right next to each other and since delta needs space between cells. So during training of delta the outer layer of the cells were eroded to
create space. So after delta cells need to be diluted again."""

def load_config(koala_config_nrIn: int, display_always_onIn: bool = None,
                local_grid_searchIn: bool = None, nfevaluationsIn : Tuple[int] = None,
                focus_methodIn: Tuple[str] = None, optimizing_methodIn: str = None,
                reconstruction_distance_lowIn: float = None, reconstruction_distance_highIn: float = None,
                reconstruction_distance_guessIn: float = None, plane_fit_orderIn: int = None,
                image_cutIn: Tuple[Tuple[int, int], Tuple[int, int]] = None, save_formatIn: str = None,
                koala_reset_frequencyIn: int = None):
    
    """
    This function laods the koala configuration and can change any configuration of the pipeline. If None are selected recommended caonfigurations are used.
    Configurations only reset to recommended configurations after restarting the python kernel.
    """
    
    # Update the configuration settings if arguments are provided
    global koala_config_nr
    koala_config_nr = koala_config_nrIn
    
    if display_always_onIn is not None:
        global display_always_on
        display_always_on = display_always_onIn
    
    if local_grid_searchIn is not None:
        global local_grid_search
        local_grid_search = local_grid_searchIn
        
    if nfevaluationsIn is not None:
        global nfevaluations
        nfevaluations = nfevaluationsIn

    if focus_methodIn is not None:
        global focus_method
        focus_method = focus_methodIn
    
    if optimizing_methodIn is not None:
        global optimizing_method
        optimizing_method = optimizing_methodIn
    
    if reconstruction_distance_lowIn is not None:
        global reconstruction_distance_low
        reconstruction_distance_low = reconstruction_distance_lowIn
    
    if reconstruction_distance_highIn is not None:
        global reconstruction_distance_high
        reconstruction_distance_high = reconstruction_distance_highIn
    
    if reconstruction_distance_guessIn is not None:
        global reconstruction_distance_guess
        reconstruction_distance_guess = reconstruction_distance_guessIn
    
    if plane_fit_orderIn is not None:
        global plane_fit_order
        plane_fit_order = plane_fit_orderIn
        
    if image_cutIn is not None:
        global image_cut
        image_cut = image_cutIn

    if save_formatIn is not None:
        global save_format
        save_format = save_formatIn
    
    if koala_reset_frequencyIn is not None:
        global koala_reset_frequency
        koala_reset_frequency = koala_reset_frequencyIn
    
    global _LOADED
    _LOADED = True

def set_image_variables(image_size_in: Tuple, px_size_in: float, laser_lambd: float):
    # hconv is hard coded and can not be changed, since Koala uses the wrong hconv
    global image_size
    image_size = image_size
    global px_size
    px_size = px_size_in
    global hconv
    hconv = laser_lambd/(2*math.pi)