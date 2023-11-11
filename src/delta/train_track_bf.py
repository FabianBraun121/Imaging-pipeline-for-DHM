# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:27:05 2023

@author: SWW-Bc20
"""
import os
import sys
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from delta.utilities import cfg as delta_cfg
from delta.model import unet_track
from delta.data import trainGenerator_track

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
import config as dhm_cfg


# Load config:
delta_cfg.load_config(presets="2D")

# Files:
training_set = Path(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training\bf_tracking_set')
base_unet_path = Path(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\models_base_delta', 'unet_pads_track.hdf5')
save_unet = str(Path(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\data\delta_assets\models', 'unet_pads_bf_track.hdf5'))




# Training parameters:
batch_size = 2
epochs = 300
steps_per_epoch = 500
validation_steps = 300
patience = 25

# Data generator parameters:
train_data_gen_args = dict(
    rotation=1,
    zoom=0.15,
    horizontal_flip=True,
    histogram_voodoo=False,
    illumination_voodoo=False,
)

# Generator init:
myGene_train = trainGenerator_track(
    batch_size,
    os.path.join(training_set, "img"),
    os.path.join(training_set, "seg"),
    os.path.join(training_set, "previmg"),
    os.path.join(training_set, "segall"),
    os.path.join(training_set, "mot_dau"),
    os.path.join(training_set, "wei"),
    train_data_gen_args,
    target_size=delta_cfg.target_size_track,
    crop_windows=delta_cfg.crop_windows,
    shift_cropbox=5,
)

# Define model:
model = unet_track(pretrained_weights=base_unet_path, input_size=delta_cfg.target_size_track + (4,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    save_unet, monitor="loss", verbose=1, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=patience)

# Train:
history = model.fit(
    myGene_train,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping],
)
