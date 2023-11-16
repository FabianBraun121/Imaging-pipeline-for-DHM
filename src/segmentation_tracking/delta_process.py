# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 10:53:29 2023

@author: SWW-Bc20
"""
import sys
import os
from skimage.morphology import binary_dilation
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
import tensorflow as tf
import numpy as np
from typing import cast, List, Union, Dict, Optional, Any, Tuple
from pathlib import Path
import tifffile
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
            filenamesindexing: int = 1       
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
            self.saving_dir = self.base_dir
        self.filenamesindexing = filenamesindexing
        
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
            
            # core sgementation and tracking
            core_models = dict()
            core_dir = str(pos_dir) + os.sep + 'bacteria_cores'
            if not os.path.exists(core_dir): 
                os.makedirs(core_dir) 
            if cfg.delta_bacteria_core == "BF":
                self._save_bf_as_delta_compatible(str(pos_dir), core_dir)
                core_models["segmentation"] = tf.keras.models.load_model(cfg.model_file_bf_core_seg, compile=False)
                core_models["tracking"] = tf.keras.models.load_model(cfg.model_file_bf_track, compile=False)
            if cfg.delta_bacteria_core == "PH":
                self._save_ph_as_delta_compatible(str(pos_dir), core_dir)
                core_models["segmentation"] = tf.keras.models.load_model(cfg.model_file_ph_core_seg, compile=False)
                core_models["tracking"] = tf.keras.models.load_model(cfg.model_file_ph_track, compile=False)
            core_xpreader = delta.utils.xpreader(core_dir)
            delta_core_position = delta.pipeline.Position(0, core_xpreader, core_models, drift_correction=False, crop_windows=True)
            frames = [f for f in range(core_xpreader.timepoints)]
            delta_core_position.preprocess(rotation_correction=False)
            delta_core_position.segment(frames=frames)
            delta_core_position.track(frames=frames)
            # features needs to be processed so label_stack gets claculated
            delta_core_position.features(frames=frames)
            
            # watershed to the full bacteria
            full_bacteria_models = dict()
            full_bacteria_dir = str(pos_dir) + os.sep + 'full_bacteria'
            if not os.path.exists(full_bacteria_dir): 
                os.makedirs(full_bacteria_dir)
            self._save_ph_as_delta_compatible(str(pos_dir), full_bacteria_dir)
            full_bacteria_models["segmentation"] = tf.keras.models.load_model(cfg.model_file_ph_full_seg, compile=False)
            full_bacteria_xpreader = delta.utils.xpreader(full_bacteria_dir)
            delta_full_position = delta.pipeline.Position(0, full_bacteria_xpreader, full_bacteria_models, drift_correction=False, crop_windows=True)
            delta_full_position.preprocess(rotation_correction=False)
            delta_full_position.segment(frames=frames)
            
            
            delta_core_position.rois[0].label_stack = self._watershed_labeled_cores_to_bacteria_outsides(delta_core_position.rois[0].label_stack,
                                                                                                         delta_full_position.rois[0].seg_stack)
            delta_core_position.rois[0].img_stack = delta_full_position.rois[0].img_stack
            delta_core_position.reader = delta_full_position.reader
            
            # Extract features:
            features = ("length", "width", "area", "perimeter", "edges")
            delta_core_position.features(frames=frames, features=features)
            
            for cell in delta_core_position.rois[0].lineage.cells:
                cell['mean_opl'] = []
                for frame in cell['frames']:
                    # cells are marked 1 higher in the label stack, because cell id starts with 0 and 0 is the background in the label stack
                    mask = delta_core_position.rois[0].label_stack[frame] == cell['id'] + 1
                    cell['mean_opl'].append((np.mean(delta_core_position.rois[0].img_stack[frame][mask])/65535*np.pi-np.pi/2) * cfg.hconv)
                cell['integrated_opl'] = list((np.array(cell['mean_opl']) * 1e-3)  * (np.array(cell['area']) * (cfg.px_size)**2)) #opl in micro meters cubed
                cell['mass'] = list((1 / 0.18) * np.array(cell['integrated_opl'])) # pico gram
            
            delta_outputs_dir = str(pos_dir) + os.sep + 'delta_outputs'
            if not os.path.exists(delta_outputs_dir): 
                os.makedirs(delta_outputs_dir)
            # Save to disk and clear memory:
            delta_core_position.save(filename= Path(delta_outputs_dir, f"Position{int(pos_name):05d}"),frames=frames)
            
            delta_full_position.clear()
            delta_core_position.clear()
    
    def _watershed_labeled_cores_to_bacteria_outsides(self, label_stack, seg_stack):
        label_stack_out = []
        for i in range(len(label_stack)):
            distance = distance_transform_edt(label_stack[i])
            # label_stack[i] = watershed(-distance, label_stack[i], mask=seg_stack[i])
            label_stack_out.append(watershed(-distance, label_stack[i], mask=seg_stack[i]))
        return label_stack_out
    
    def _save_ph_as_delta_compatible(self, pos_dir, saving_pos_dir):
        ph_fnames = [ph for ph in os.listdir(pos_dir) if 'PH' in ph]
        for i, ph_fname in enumerate(ph_fnames):
            ph_image = tifffile.imread(pos_dir + os.sep + ph_fname)
            ph_scaled = (((ph_image + np.pi/2) / np.pi) * 65535).astype(np.uint16)
            fname = saving_pos_dir + os.sep + f"pos{str(self.filenamesindexing).zfill(5)}cha{self.filenamesindexing}fra{str(i+self.filenamesindexing).zfill(5)}.tif"
            tifffile.imwrite(fname, ph_scaled)
    
    def _save_bf_as_delta_compatible(self, pos_dir, saving_pos_dir):
        bf_fnames = [bf for bf in os.listdir(pos_dir) if 'BF' in bf]
        for i, bf_fname in enumerate(bf_fnames):
            bf_image = tifffile.imread(pos_dir + os.sep + bf_fname)
            bf_scaled = ((bf_image - bf_image.min())/(bf_image.max() - bf_image.min()) * 65535).astype(np.uint16)
            fname = saving_pos_dir + os.sep + f"pos{str(self.filenamesindexing).zfill(5)}cha{self.filenamesindexing}fra{str(i+self.filenamesindexing).zfill(5)}.tif"
            tifffile.imwrite(fname, bf_scaled)

