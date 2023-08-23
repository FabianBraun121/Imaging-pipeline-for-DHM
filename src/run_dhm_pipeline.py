# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))

import spatial_averaging as sa
import segmentation_tracking as st
import config

koala_config_nr = 234
restrict_positions = None   # slice
restrict_timesteps = range(0, 100, 25) # range
select_recon_rectangle = False
select_image_roi = False
# base_dir = sa.utilities.Open_Directory(r'Q:\SomethingFun' , "Open a scanning directory")
base_dir = r'F:/F3_20230406/2023-04-06 11-07-24'

################# spatial averaging ####################################
config.load_config(koala_config_nrIn=koala_config_nr)
pipe = sa.pipeline_test_images.Pipeline(base_dir=base_dir, restrict_positions=restrict_positions,
                            restrict_timesteps=restrict_timesteps)
if select_recon_rectangle:
    pipe.select_positions_recon_rectangle(same_for_all_pos = False)
if select_image_roi:
    pipe.select_positions_image_roi(same_for_all_pos = False)
pipe.process()

# ################# segmentation and tracking ############################

# restrict_positions = slice(0,1)
# pipe_delta = st.delta_process.Delta_process(pipe.saving_dir, restrict_positions=restrict_positions)
# #pipe_delta = st.delta_process.Delta_process(r'F:/C11_20230217/2023-02-17 11-13-34 phase averages', restrict_positions=restrict_positions)
# pipe_delta.process()
