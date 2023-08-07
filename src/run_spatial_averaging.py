# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src')

import spatial_averaging as sa
import segmentation_tracking as st
import config

koala_config_nr = 222
restrict_positions = slice(1,2)    # slice
restrict_timesteps = range(0,1)    # range
select_recon_rectangle = False
select_image_roi = False
base_dir = sa.utilities.Open_Directory(r'Q:\SomethingFun' , "Open a scanning directory")

################# spatial averaging ####################################
config.load_config(koala_config_nrIn=koala_config_nr)
pipe = sa.pipeline.Pipeline(base_dir=base_dir, restrict_positions=restrict_positions,
                            restrict_timesteps=restrict_timesteps, )
if select_recon_rectangle:
    pipe.select_positions_recon_rectangle(same_for_all_pos = False)
if select_image_roi:
    pipe.select_positions_image_roi(same_for_all_pos = False)
pipe.process()

################# segmentation and tracking ############################
pipe_delta = st.delta_process.Delta_process(pipe.saving_dir, restrict_positions=restrict_positions)
pipe_delta.process()
