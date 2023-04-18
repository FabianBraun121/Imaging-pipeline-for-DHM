# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src')

import spatial_averaging as sa

#base_dir = sa.utilities.Open_Directory(r'Q:\SomethingFun' , "Open a scanning directory")
base_dir = 'F:/E10_20230216/2023-02-16 12-16-39'
sa.config.load_config(koala_config_nr=218, display_always_on=True)
pipe = sa.pipeline.Pipeline(base_dir=base_dir, restrict_locations=slice(1,8,1), restrict_timesteps=slice(0,171,1))
#pipe.select_locations_roi()
pipe.process()