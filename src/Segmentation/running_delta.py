# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import delta
#%%

delta.config.load_config()

#%%
reader = delta.utilities.xpreader('F:/C11_20230217/test_float',
                                  prototype='F:/C11_20230217/test_float/pos%02d_chan%02d_time%03d.tif',
                                  fileorder='pct',
                                  filenamesindexing=1)
#%%
processor = delta.pipeline.Pipeline(reader)
#%%
processor.process()
