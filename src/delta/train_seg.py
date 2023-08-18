"""
Created on Mon Apr 10 13:04:19 2023

@author: SWW-Bc20
"""
import os

import sys
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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
savefile = Path(assets_path, 'models', 'unet_pads_seg.hdf5')

# Training parameters:
batch_size = 1
epochs = 300
steps_per_epoch = 1000
validation_steps = 50
patience = 50

# Data generator parameters:
train_data_gen_args = dict(
    rotation=2,
    rotations_90d=True,
    zoom=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=True,
    gaussian_noise=0.003,
    gaussian_blur=0.1,
    # elastic_deformation={'sigma': 50, 'points': 3},
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

myGene_val = trainGenerator_seg(
    batch_size,
    os.path.join(validation_set, "img"),
    os.path.join(validation_set, "seg"),
    os.path.join(validation_set, "wei"),
    target_size=delta_cfg.target_size_seg,
    crop_windows=delta_cfg.crop_windows,
)

# Define model:
model = unet_seg(pretrained_weights=savefile, input_size=delta_cfg.target_size_seg + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="val_loss", verbose=2, save_best_only=True
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

