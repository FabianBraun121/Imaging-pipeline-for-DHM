# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.append(r'C:\Users\SWW-Bc20\Documents\GitHub\delta-main')
import delta
#%%

delta.config.load_config(presets='2D',config_level='local')

#%%
reader = delta.utilities.xpreader('F:/C11_20230217/test_float',
                                  prototype='F:/C11_20230217/test_float/pos%02d_chan%02d_time%03d.tif',
                                  fileorder='pct',
                                  filenamesindexing=1)
#%%
processor = delta.pipeline.Pipeline(reader)
#%%
processor.process()
