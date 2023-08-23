# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:02:45 2023

@author: SWW-Bc20
"""
import os

import sys
from pathlib import Path
import pickle

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import gc

from delta.utilities import cfg as delta_cfg
from delta.model import unet_seg
from delta.data import trainGenerator_seg

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
import config as dhm_cfg

# Load config:
delta_cfg.load_config(presets="2D")

assets_path = dhm_cfg.delta_assets_path
# Files:
training_set = Path(assets_path, 'trainingsets', '2D', 'training', 'segmentation_set')
validation_set = Path(assets_path, 'trainingsets', '2D', 'validation', 'segmentation_set')
unet_pretrained = Path(assets_path, 'models', 'unet_pads_seg.hdf5')

# Training parameters:
batch_size = 1
epochs = 300
steps_per_epoch = 500
validation_steps = 200
patience = 25

noise = [0.0024, 0.012, 0.06]
blur = [0.08, 0.4, 2]
ed_sigma = [0, 10, 50]

for i in range(len(ed_sigma)):
    for j in range(len(blur)):
        
        # Data generator parameters:
        train_data_gen_args = dict(
            rotation=2,
            rotations_90d=True,
            zoom=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            illumination_voodoo=True,
            gaussian_noise=noise[j],
            gaussian_blur=blur[j],
        )
        if ed_sigma[i] != 0:
            train_data_gen_args['elastic_deformation']={'sigma': ed_sigma[i], 'points': 3}
        
        augmentation = f'noise_{noise[j]}_blur_{blur[j]}_ed_sigma_{ed_sigma[i]}'
        save_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\tests\delta\data_augmentation' + os.sep + augmentation
        
        # Generator init:
        myGene_train = trainGenerator_seg(
            batch_size,
            os.path.join(training_set, "img"),
            os.path.join(training_set, "seg"),
            os.path.join(training_set, "wei"),
            augment_params=train_data_gen_args,
            target_size=delta_cfg.target_size_seg,
            crop_windows=delta_cfg.crop_windows,
        )
        
        myGene_val = trainGenerator_seg(
            batch_size,
            os.path.join(validation_set, "img"),
            os.path.join(validation_set, "seg"),
            os.path.join(validation_set, "wei"),
            target_size=delta_cfg.target_size_seg,
            crop_windows=delta_cfg.crop_windows,
        )
        
        # Define model:
        model = unet_seg(pretrained_weights=unet_pretrained, input_size=delta_cfg.target_size_seg + (1,))
        model.summary()
        
        # Callbacks:
        model_checkpoint = ModelCheckpoint(
            save_path+'.hdf5', monitor="val_loss", verbose=2, save_best_only=True
        )
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=patience)
        
        # Train:
        history = model.fit(
            myGene_train,
            validation_data=myGene_val,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[model_checkpoint, early_stopping],
        )
        
        with open(save_path+'.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
        K.clear_session()
        del model
        del myGene_train
        del myGene_val
        del model_checkpoint
        del early_stopping
        del history
        gc.collect()
