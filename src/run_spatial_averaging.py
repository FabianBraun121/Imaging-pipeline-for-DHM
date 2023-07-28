# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src')

import spatial_averaging as sa

# base_dir = sa.utilities.Open_Directory(r'Q:\SomethingFun' , "Open a scanning directory")
base_dir = r'F:\F9_20230228\2023-02-28 12-21-37'
sa.config.load_config(koala_config_nr=221, display_always_on=True)
pipe = sa.pipeline.Pipeline(base_dir=base_dir, restrict_locations=slice(0,1), restrict_timesteps=range(3))
pipe.process()
    