# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 10:53:29 2023

@author: SWW-Bc20
"""
import sys
import os
from skimage.morphology import binary_dilation
import numpy as np
from typing import cast, List, Union, Dict, Optional, Any, Tuple
from pathlib import Path
import delta

sys.path.append("..")
import config as cfg


class Delta_process:
    def __init__(
            self,
            base_dir: Union[str, Path],
            saving_dir: Union[str, Path] = None,
            restrict_positions: slice = None,
            ):
    
        self.base_dir: Path = Path(base_dir)
        self.data_file_path: Path = None
        self.restrict_positions: slice = restrict_positions
        self.positions = self._positions()
        if saving_dir is not None:
            self.saving_dir: Path = Path(saving_dir)
        else:
            self.saving_dir = None
    
    def _positions(self):
        positions = [d for d in os.listdir(self.base_dir) if os.path.isdir(Path(self.base_dir,d))]
        if self.restrict_positions is None:
            return positions
        else:
            return positions[self.restrict_positions]
        
    def process(self):
        delta.config.load_config(presets="2D")
        
        for pos_name in self.positions:
            pos_dir = Path(self.base_dir,pos_name)
            if self.saving_dir is not None:
                saving_dir_pos = Path(self.saving_dir, pos_name)
            else:
                saving_dir_pos = None
            xpreader = delta.utils.xpreader(pos_dir)
            pipe = delta.pipeline.Pipeline(xpreader, resfolder=saving_dir_pos)
            pipe.process(clear=False)
            
            frames = [f for f in range(xpreader.timepoints)]
            features = ("length", "width", "area", "perimeter", "edges", "fluo1")
            
            for f in frames:
                pipe.positions[0].rois[0].label_stack[f] = self._dilute_cells(pipe.positions[0].rois[0].label_stack[f], cfg.dilute_cells)
            # Extract features:
            pipe.positions[0].features(frames=frames, features=features)
            
            for cell in pipe.positions[0].rois[0].lineage.cells:
                cell['mean_opl'] = list((np.array(cell['fluo1'])/65535*np.pi-np.pi/2)*794/(2*np.pi))
                cell['integrated_opl'] = list(np.array(cell['mean_opl']) * cfg.hconv*1e6 * (cfg.px_size*1e6)**2)
            
            # Save to disk and clear memory:
            pipe.positions[0].save(
                filename= Path(pipe.resfolder, f"Position{int(pos_name):06d}"),
                frames=frames,
                save_format=pipe.save_format,
            )
            
            [os.remove(Path(pipe.resfolder,filename)) for filename in os.listdir(pipe.resfolder) if filename.startswith("Position000000")]
            
            del pipe
    
    def _dilute_cells(self, label_stackIn, n):
        label_stack = label_stackIn.copy()
        for i in range(n):
            for cell_label in np.unique(label_stack):
                # Skip the background region (value of 0)
                if cell_label == 0:
                    continue
                
                label_mask = (label_stack == cell_label)
                dilated_mask = binary_dilation(label_mask)
                free_dilated_mask = (label_stack == 0) & dilated_mask
                label_stack[free_dilated_mask] = cell_label
        return label_stack


