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
import re

sys.path.append("..")
import config as cfg


class Delta_process:
    """
        Some small changes and additions are made to the DeLTA pipeline, so it works seemlessly with 
        phase difference images and the spatial averaging pipeline.
    """    
    def __init__(
            self,
            base_dir: Union[str, Path],
            saving_dir: Union[str, Path] = None,
            restrict_positions: slice = None,
            ):
        """
        Parameters
        ----------
        base_dir : Union[str, Path]
            Directory where the various positions folders with the time-lapse images is.
        saving_dir : Union[str, Path], optional
            Directory where the results are saved. The default is None.
        restrict_positions : slice, optional
            Restrict the positions processed. The default is None.

        Returns
        -------
        None.

        """
        self.base_dir: Path = Path(base_dir)
        "Directory of the various positions folders"
        self.data_file_path: Path = None
        "Path for a processing data file. Not in use at the moment."
        self.restrict_positions: slice = restrict_positions
        "Restrict the positions processed"
        self.positions = self._positions()
        if saving_dir is not None:
            self.saving_dir: Path = Path(saving_dir)
        else:
            self.saving_dir = None
    
    def _positions(self) -> List[str]:
        "returns a list of th positions processed."
        positions = [d for d in os.listdir(self.base_dir) if os.path.isdir(Path(self.base_dir,d))]
        if self.restrict_positions is None:
            return positions
        else:
            return positions[self.restrict_positions]
        
    def process(self):
        """processes the positions. First with DeLTA. Then bacteria are enlarged.
        Mean opl, integrated opl and mass are calculated. Newly calculated data is saved."""
        delta.config.load_config(presets="2D")
        
        for pos_name in self.positions:
            pos_dir = Path(self.base_dir,pos_name)
            if self.saving_dir is not None:
                saving_dir_pos = Path(self.saving_dir, pos_name)
            else:
                saving_dir_pos = None
            self._rename_to_delta(pos_dir)
            xpreader = delta.utils.xpreader(pos_dir)
            pipe = delta.pipeline.Pipeline(xpreader, resfolder=saving_dir_pos)
            pipe.process(clear=False)
            
            frames = [f for f in range(xpreader.timepoints)]
            features = ("length", "width", "area", "perimeter", "edges", "fluo1")
            
            for f in frames:
                pipe.positions[0].rois[0].label_stack[f] = self._dilute_cores_to_cells(pipe.positions[0].rois[0].label_stack[f],
                                                                                       pipe.positions[0].rois[0].img_stack[f],
                                                                                       cfg.dilute_cells)
            # Extract features:
            pipe.positions[0].features(frames=frames, features=features)
            
            for cell in pipe.positions[0].rois[0].lineage.cells:
                cell['mean_opl'] = list((np.array(cell['fluo1'])/65535*np.pi-np.pi/2) * cfg.hconv*1e9) # mean_opl in nanometers
                cell['integrated_opl'] = list((np.array(cell['mean_opl']) * 1e-3)  * (np.array(cell['area']) * (cfg.px_size*1e6)**2)) #opl in micro meters cubed
                cell['mass'] = list((1 / 0.18) * np.array(cell['integrated_opl'])) # pico gram
            
            # Save to disk and clear memory:
            pipe.positions[0].save(
                filename= Path(pipe.resfolder, f"Position{int(pos_name):05d}"),
                frames=frames,
                save_format=pipe.save_format,
            )
            
            [os.remove(Path(pipe.resfolder,filename)) for filename in os.listdir(pipe.resfolder) if filename.startswith("Position000000")]
            
            del pipe
    
    def _dilute_cells(self, labelsIn, n):
        "dilutes the cells n times"
        labels = labelsIn.copy()
        for i in range(n):
            for cell_label in np.unique(labels):
                # Skip the background region (value of 0)
                if cell_label == 0:
                    continue
                
                labels_mask = (labels == cell_label)
                dilated_mask = binary_dilation(labels_mask)
                free_dilated_mask = (labels == 0) & dilated_mask
                labels[free_dilated_mask] = cell_label
        return labels
    
    def _dilute_cores_to_cells(self, labelsIn, img, n):
        "dilutes the core to a cell, first three dilutions are only on high image pixel values. Dilutions after are fixed."
        labels = labelsIn.copy()
        img_cut_off = np.percentile(img, 95)
        for i in range(n):
            for cell_label in np.unique(labels):
                # Skip the background region (value of 0)
                if cell_label == 0:
                    continue
                
                mask = (labels == cell_label)
                dilated_mask = binary_dilation(mask)
                boundary = dilated_mask & ~mask
                boundary = (labels == 0) & boundary
                if i<3:
                    boundary = (img_cut_off < img) & boundary
                labels[boundary] = cell_label
        return labels
    
    def _rename_from_delta(self, pos_dir):
        "images in a folder of Delta with the same position are always number 1. Function changes to true position name."
        for old_filename in os.listdir(pos_dir):
            new_filename = re.sub('pos00001', f'pos{pos_dir.name}', old_filename)
            os.rename(Path(pos_dir,old_filename), Path(pos_dir,new_filename))
    
    def _rename_to_delta(self, pos_dir):
        "change pos name to pos00001"
        for old_filename in os.listdir(pos_dir):
            new_filename = re.sub(r'pos(\d+)', 'pos00001', old_filename)
            os.rename(Path(pos_dir,old_filename), Path(pos_dir,new_filename))


