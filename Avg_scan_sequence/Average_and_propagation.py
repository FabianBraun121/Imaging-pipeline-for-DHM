# -*- coding: utf-8 -*-
"""
Created on 2023.01.23
Modified 2023.02.09 : autofocus implementation
Modified 2023.03.22 : Add computation shift with unwrapped phase

@author: tcolomb
"""
import ProcessClass, ProcessPropClass

default_dir = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM'


"""
Procedures to perform
"""
perform_avg = False
perform_propagation = True

"""
Parameters for the averaging
"""
#use_amplitude, if amplitude does not exist, message, if False, use the phase
use_amplitude_for_shift = False
shift_timeline = False #superpose data of the timeline data
Use_ROI = False #use region of interest to measure the shift (a ROI for each well)
unwrap = True #Perform path following unwrap on resulted averaged phase image
UseScanValue = False #Use the stage position to perform superposition instead of shift algorithm
use_background = False# works only if the displacement is quite large and to use only if there are lot of noise
use_unwrap_for_shifting = True # is not used if amplitude is the critera, else it unwrap the reference image and the new image to shift

"""
Parameters for the propagation
"""
d = -0.0 #reconstruction distance in cm for all wells d = d_config-d_used for the acquisition (d_used=0 if reconstruction perform in holo plane)
define_rec_dist = True #define manually or with autofocus the reconstruction distance for all wells 
autofocus = False #perform autofocus for the first averaged wavefront of each well
autofocus_each_time_line = False#if autofocus=True, perform autofocus for all timeline
focusMethod = 1 #0=std, 1= Louis method
range_focus_um = 20 #approximative range in micron of displacement of the sample to define the range of autofocus
ROI = 1000 #usual ROIxROI or reconstructed data, None do not perform ROI
ROI = None

"""
DHM parameters: constant that have to be defined from Koala (camera model, wavelength)
"""
CCD_px_size_um = 0.13 #pixel size of the CCD DHM camera (has to be the right one for autofocus !)
lambda_nm = 794 #wavelength in nanometer of the DHM (has to be the right one for autofocus!)


"""
Process
if perform propagation, do not apply unwrap on the average because unnecessary, will be done on the propagated wavefronts
"""
if perform_propagation:
    unwrap_avg = False
else:
    unwrap_avg = unwrap

if perform_avg:
    #construct class for processing
    complex_averaging_class = ProcessClass.complex_averaging(shift_timeline, Use_ROI, use_amplitude_for_shift, UseScanValue, unwrap_avg, use_background, use_unwrap_for_shifting)
    #Choose directory
    complex_averaging_class.Open_Directory(default_dir, "Open a scanning directory")
    ##Perform averaging (if data (phase bin) are missing stop)
    print("Perform averaging")
    if complex_averaging_class.set_directory():
        complex_averaging_class.process_all_wells()
    else:
        print("Error: not possible to perform propagation")
        perform_propagation = False


if perform_propagation:
    print("Perform propagation")
    d_init_cm = d
    Propagation = ProcessPropClass.propagation_avg(lambda_nm*1e-9, d*1e-2,CCD_px_size_um*1e-6,unwrap, define_rec_dist, ROI, autofocus, autofocus_each_time_line, d_init_cm, range_focus_um, focusMethod)
    if perform_avg:
        directory_to_save = complex_averaging_class.get_directory_to_save()      
        directory = Propagation.Set_Directory_input(directory_to_save)
    else:
        Propagation.Open_Directory(default_dir, "Open an averaging directory")
    if Propagation.set_directory():
        Propagation.process_all_wells()

