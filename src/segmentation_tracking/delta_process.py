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
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
from pathlib import Path
import tifffile
import delta
import glob
import re
import pandas as pd

sys.path.append("..")
from src.config import Config


class Delta_process:
    """
        Some small changes and additions are made to the DeLTA pipeline, so it works seemlessly with 
        phase difference images and the spatial averaging pipeline.
    """    
    def __init__(
            self,
            config: Config,
            base_dir: Union[str, Path] = None,
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
        self.cfg: Config = config
        "All settings are saved in this config class"
        self.cfg.set_config_setting('delta_base_dir', self._base_dir(base_dir))
        "Directory of the various positions folders"
        self.cfg.set_config_setting('delta_saving_dir', self._saving_dir(saving_dir))
        self.data_file_path: Path = None
        "Path for a processing data file. Not in use at the moment."
        self.restrict_positions: slice = restrict_positions
        "Restrict the positions processed"
        self.positions = self._positions()
        self.filenamesindexing = filenamesindexing
        
    def _base_dir(self, base_dir):
        if base_dir is not None:
            return Path(base_dir)
        else:
            return Path(self.cfg.get_config_setting('saving_dir'))
        
    def _saving_dir(self, saving_dir):
        if saving_dir is not None:
            return Path(saving_dir)
        else:
            return Path(self.cfg.get_config_setting('delta_base_dir'))
    
    def _positions(self) -> List[str]:
        "returns a list of th positions processed."
        delta_base_dir = self.cfg.get_config_setting('delta_base_dir')
        positions = [d for d in os.listdir(delta_base_dir) if os.path.isdir(Path(delta_base_dir,d))]
        if self.restrict_positions is None:
            return positions
        else:
            return positions[self.restrict_positions]
        
    def process(self):
        if self.cfg.get_config_setting('save_as_bulk'):
            self._process_seg()
        else:
            self._process_seg_and_track()
    
    def _process_seg(self):
        delta.config.load_config(presets="2D")
        
        saving_delta_dir = self.cfg.get_config_setting('delta_saving_dir')
        if not os.path.exists(saving_delta_dir): 
            os.makedirs(saving_delta_dir)
        
        ph_folder = Path(self.cfg.get_config_setting('delta_saving_dir'), 'PH')
        core_folder =  Path(self.cfg.get_config_setting('delta_saving_dir'), self.cfg.get_config_setting('delta_bacteria_core'))
        delta_ph_folder = Path(self.cfg.get_config_setting('delta_saving_dir'), 'delta_PH')
        if not os.path.exists(delta_ph_folder): 
            os.makedirs(delta_ph_folder)
        files_list = []
        for fname in sorted(os.listdir(ph_folder)):
            core_folder_list = [f[:-7] for f in os.listdir(core_folder)]
            if fname[:-7] in core_folder_list and fname.endswith('.tif'):
                self._read_rescale_save_ph_image(Path(ph_folder, fname), Path(delta_ph_folder, fname))
                files_list.append(fname[:-7])
        
        if self.cfg.get_config_setting('delta_bacteria_core') == 'PH':
            core_folder = Path(self.cfg.get_config_setting('delta_saving_dir'), 'delta_PH')
        
        core_seg_weights = self.cfg.get_config_setting(f'model_file_{self.cfg.get_config_setting("delta_bacteria_core").lower()}_core_seg')
        core_seg = self._process_seg_images(core_folder, files_list, self.cfg.get_config_setting('delta_bacteria_core'), core_seg_weights)
        label_stack = []
        for i in range(core_seg.shape[0]):
            labeled_array, _ = label(core_seg[i])
            label_stack.append(labeled_array)
        
        full_seg_weights = self.cfg.get_config_setting('model_file_ph_full_seg')
        full_seg = self._process_seg_images(delta_ph_folder, files_list, 'PH', full_seg_weights)
        
        labeled_full_ph_seg = self._watershed_labeled_cores_to_bacteria_outsides(label_stack, list(full_seg))
        data = []
        seg_folder = Path(self.cfg.get_config_setting('delta_saving_dir'), 'SEG')
        if not os.path.exists(seg_folder): 
            os.makedirs(seg_folder)
            
        seg__folder = Path(self.cfg.get_config_setting('delta_saving_dir'), 'SEG_')
        if not os.path.exists(seg__folder): 
            os.makedirs(seg__folder)
        
        for i in range(len(files_list)):
            ph_image = tifffile.imread(Path(ph_folder, f'{files_list[i]}_PH.tif'))
            regions = regionprops(labeled_full_ph_seg[i])
            tifffile.imwrite(Path(seg__folder, f'{files_list[i]}.tif'), label_stack[i])
            tifffile.imwrite(Path(seg_folder, f'{files_list[i]}.tif'), labeled_full_ph_seg[i])
            for props in regions:
                mean_opl_nm = np.mean(ph_image[labeled_full_ph_seg[i]==props.label]) * self.cfg.get_config_setting('hconv')
                integrated_opl_μm_cubed = mean_opl_nm * 1e-3 * props.area * self.cfg.get_config_setting('px_size')**2
                mass_fg = integrated_opl_μm_cubed / 0.18 * 1000
                row = {
                    'image_name': files_list[i],
                    'label': props.label,
                    'centroid': props.centroid,
                    'full_ph_area_px': props.area,
                    'full_ph_area_μm_sq': props.area * self.cfg.get_config_setting('px_size')**2,
                    'full_ph_length_px': props.major_axis_length,
                    'full_ph_length_μm': props.major_axis_length * self.cfg.get_config_setting('px_size'),
                    'full_ph_width_px': props.minor_axis_length,
                    'full_ph_width_μm': props.minor_axis_length * self.cfg.get_config_setting('px_size'),
                    'mean_opl_nm': mean_opl_nm,
                    'integrated_opl_μm_cubed': integrated_opl_μm_cubed,
                    'mass_fg': mass_fg
                    }
                seg_props = regionprops((label_stack[i]==props.label).astype(np.uint8))[0]
                row['seg_area_px'] = seg_props.area
                row['seg_area_μm_sq'] = seg_props.area * self.cfg.get_config_setting('px_size')**2
                row['seg_length_px'] = seg_props.major_axis_length
                row['seg_length_μm'] = seg_props.major_axis_length * self.cfg.get_config_setting('px_size')
                row['seg_width_px'] = seg_props.minor_axis_length
                row['seg_width_μm'] = seg_props.minor_axis_length * self.cfg.get_config_setting('px_size')
                data.append(row)
        pd.DataFrame(data).to_csv(Path(saving_delta_dir, 'bacteria_data.csv'), index=False)
            
    def _process_seg_images(self, folder, files_list, image_type, model_weights):
        model = delta.model.unet_seg(input_size=delta.config.target_size_seg + (1,))
        model.load_weights(model_weights)

        to_process = [str(Path(folder, f'{i}_{image_type}.tif')) for i in files_list]
        predGene = delta.data.predictGenerator_seg(
            folder,
            files_list=to_process,
            target_size=delta.config.target_size_seg,
            crop=delta.config.crop_windows)
        
        img = delta.data.readreshape(
            os.path.join(folder, to_process[0]),
            target_size=delta.config.target_size_seg,
            crop=True,
        )
        # Create array to store predictions
        results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))
        # Crop, segment, stitch and store predictions in results
        for i in range(len(to_process)):
            # Crop each frame into overlapping windows:
            windows, loc_y, loc_x = delta.utilities.create_windows(
                next(predGene)[0, :, :], target_size=delta.config.target_size_seg
            )
            # We have to play around with tensor dimensions to conform to
            # tensorflow's functions:
            windows = windows[:, :, :, np.newaxis]
            # Predictions:
            pred = model.predict(windows, verbose=1, steps=windows.shape[0])
            # Stich prediction frames back together:
            pred = delta.utilities.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
            pred = pred[np.newaxis, :, :, np.newaxis]  # Mess around with dims

            results[i] = pred
        
        results = delta.data.postprocess(results, crop=delta.config.crop_windows)
        return results

    
    def _process_seg_and_track(self):
        """processes the positions. First with DeLTA. Then bacteria are enlarged.
        Mean opl, integrated opl and mass are calculated. Newly calculated data is saved."""
        delta.config.load_config(presets="2D")
        
        for pos_name in self.positions:
            pos_dir = Path(self.cfg.get_config_setting('delta_base_dir'), pos_name)
            
            # core sgementation and tracking
            core_models = dict()
            core_dir = str(pos_dir) + os.sep + 'bacteria_cores'
            if not os.path.exists(core_dir): 
                os.makedirs(core_dir) 
            if self.cfg.get_config_setting('delta_bacteria_core') == "BF":
                self._save_bf_as_delta_compatible(str(pos_dir), core_dir)
                core_models["segmentation"] = tf.keras.models.load_model(self.cfg.get_config_setting('model_file_bf_core_seg'), compile=False)
                core_models["tracking"] = tf.keras.models.load_model(self.cfg.get_config_setting('model_file_bf_track'), compile=False)
            if self.cfg.get_config_setting('delta_bacteria_core') == "PH":
                self._save_ph_as_delta_compatible(str(pos_dir), core_dir)
                core_models["segmentation"] = tf.keras.models.load_model(self.cfg.get_config_setting('model_file_ph_core_seg'), compile=False)
                core_models["tracking"] = tf.keras.models.load_model(self.cfg.get_config_setting('model_file_ph_track'), compile=False)
            core_xpreader = delta.utils.xpreader(core_dir)
            delta_core_position = delta.pipeline.Position(0, core_xpreader, core_models, drift_correction=False, crop_windows=True)
            frames = [f for f in range(core_xpreader.timepoints)]
            delta_core_position.preprocess(rotation_correction=False)
            delta_core_position.segment(frames=frames)
            delta_core_position.track(frames=frames)
            # features needs to be processed so label_stack gets claculated
            delta_core_position.features(frames=frames)
            
            # Extract seg features:
            features = ("length", "width", "area", "perimeter", "edges")
            delta_core_position.features(frames=frames, features=features)
            for cell in delta_core_position.rois[0].lineage.cells:
                for feature in features:
                    cell[f'seg_{feature}'] = cell[feature]
            
            # watershed to the full bacteria
            full_bacteria_models = dict()
            full_bacteria_dir = str(pos_dir) + os.sep + 'full_bacteria'
            if not os.path.exists(full_bacteria_dir): 
                os.makedirs(full_bacteria_dir)
            self._save_ph_as_delta_compatible(str(pos_dir), full_bacteria_dir)
            full_bacteria_models["segmentation"] = tf.keras.models.load_model(self.cfg.get_config_setting('model_file_ph_full_seg'), compile=False)
            full_bacteria_xpreader = delta.utils.xpreader(full_bacteria_dir)
            delta_full_position = delta.pipeline.Position(0, full_bacteria_xpreader, full_bacteria_models, drift_correction=False, crop_windows=True)
            delta_full_position.preprocess(rotation_correction=False)
            delta_full_position.segment(frames=frames)
            
            # replace seg stacks and reader with full phase stacks and reader
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
                    cell['mean_opl'].append((np.mean(delta_core_position.rois[0].img_stack[frame][mask])/65535*np.pi-np.pi/2) * self.cfg.get_config_setting('hconv'))
                cell['integrated_opl'] = list((np.array(cell['mean_opl']) * 1e-3)  * (np.array(cell['area']) * (self.cfg.get_config_setting('px_size'))**2)) #opl in micro meters cubed
                cell['mass'] = list((1 / 0.18) * np.array(cell['integrated_opl'])) # femto gram
            
            delta_outputs_dir = str(Path(self.cfg.get_config_setting('delta_saving_dir'), pos_name)) + os.sep + 'delta_outputs'
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
            label_stack_out.append(watershed(-distance, label_stack[i], mask=seg_stack[i]))
        return label_stack_out
    
    def _save_ph_as_delta_compatible(self, pos_dir, saving_pos_dir):
        ph_fnames = [ph for ph in os.listdir(pos_dir) if 'PH' in ph]
        for i, ph_fname in enumerate(ph_fnames):
            fname_out = saving_pos_dir + os.sep + f"pos{str(self.filenamesindexing).zfill(5)}cha{self.filenamesindexing}fra{str(i+self.filenamesindexing).zfill(5)}.tif"
            self._read_rescale_save_ph_image(pos_dir + os.sep + ph_fname, fname_out)
    
    def _read_rescale_save_ph_image(self, fpath_in, fpath_out):
        ph_image = tifffile.imread(fpath_in)
        ph_scaled = (((ph_image + np.pi/2) / np.pi) * 65535).astype(np.uint16)
        tifffile.imwrite(fpath_out, ph_scaled)
    
    def _save_bf_as_delta_compatible(self, pos_dir, saving_pos_dir):
        bf_fnames = [bf for bf in os.listdir(pos_dir) if 'BF' in bf]
        for i, bf_fname in enumerate(bf_fnames):
            bf_image = tifffile.imread(pos_dir + os.sep + bf_fname)
            bf_scaled = ((bf_image - bf_image.min())/(bf_image.max() - bf_image.min()) * 65535).astype(np.uint16)
            fname = saving_pos_dir + os.sep + f"pos{str(self.filenamesindexing).zfill(5)}cha{self.filenamesindexing}fra{str(i+self.filenamesindexing).zfill(5)}.tif"
            tifffile.imwrite(fname, bf_scaled)

