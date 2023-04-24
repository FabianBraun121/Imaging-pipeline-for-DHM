# -*- coding: utf-8 -*-
"""
Created on 2023.02.28

@author: tcolomb
"""
import ProcessClass, ProcessPropClass

default_dir = r'C:\Users\Tristan Colomb\Desktop\test_avg_from_timestamps\(B) 2023-01-17.2x3'


"""
Procedures to perform
"""
perform_avg = True
perform_propagation = True




"""
Process
if perform propagation, do not apply unwrap on the average because unnecessary, will be done on the propagated wavefronts
"""

if perform_avg:
    #construct class for processing
    complex_averaging_class = ProcessClass.complex_averaging()
    #Choose directory
    complex_averaging_class.Open_Directory(default_dir, "Open a scanning directory")
    ##Perform averaging (if data (phase bin) are missing stop)
    print("Perform averaging")
    if complex_averaging_class.set_directory():
        ph_corrupted, amp_corrupted = complex_averaging_class.test_data()
        print(ph_corrupted)
        print(amp_corrupted)


#if perform_propagation:
#    print("Perform propagation")
#    d_init_cm = d
#    Propagation = ProcessPropClass.propagation_avg(lambda_nm*1e-9, d*1e-2,CCD_px_size_um*1e-6,unwrap, define_rec_dist, ROI, autofocus, autofocus_each_time_line, d_init_cm, range_focus_um, focusMethod)
#    if perform_avg:
#        directory_to_save = complex_averaging_class.get_directory_to_save()      
#        directory = Propagation.Set_Directory_input(directory_to_save)
#    else:
#        Propagation.Open_Directory(default_dir, "Open an averaging directory")
#    if Propagation.set_directory():
#        Propagation.process_all_wells()




