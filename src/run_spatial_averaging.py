# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import numpy as np
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src')

import spatial_averaging as sa

base_dir = sa.utilities.Open_Directory(r'Q:\SomethingFun' , "Open a scanning directory")
sa.config.load_config(koala_config_nr=244, display_always_on=False)
pipe = sa.pipeline.Pipeline(base_dir=base_dir)
pipe.process()
