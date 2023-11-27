# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))

import spatial_averaging as sa
import segmentation_tracking as st
from config import Config
from gui import run_gui

config = Config()
run_gui(config)


################# spatial averaging ####################################
pipe = sa.pipeline.Pipeline(config)
pipe.process()

################# segmentation and tracking ####################################
pipe_delta = st.delta_process.Delta_process(config)
pipe_delta.process()