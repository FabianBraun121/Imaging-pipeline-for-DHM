# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:36:56 2023

@author: SWW-Bc20
"""
import os

import sys
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from delta.utilities import cfg as delta_cfg
from delta.model import unet_seg
from delta.data import trainGenerator_seg

# Load config:
delta_cfg.load_config(presets="2D")

# Files:
training_set = Path(r'D:\data\full_ph_segmentation\segmentation_set')
base_unet_path = Path(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\models_base_delta', 'unet_pads_seg.hdf5')
save_unet = str(Path(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\models', 'unet_pads_full_ph_seg.hdf5'))


# Training parameters:
batch_size = 1
epochs = 300
steps_per_epoch = 300
validation_steps = 300
patience = 50

# Data generator parameters:
train_data_gen_args = dict(
    rotation=2,
    rotations_90d=True,
    zoom=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=False,
    gaussian_noise=0.06,
    gaussian_blur=2,
    elastic_deformation={'sigma': 50, 'points': 3},
)


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

# Define model:
model = unet_seg(pretrained_weights=base_unet_path, input_size=delta_cfg.target_size_seg + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    save_unet, monitor="loss", verbose=2, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=2, patience=patience)

# Train:
history = model.fit(
    myGene_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping],
)
