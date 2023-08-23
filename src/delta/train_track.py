"""
Created on Mon Apr 10 13:04:19 2023

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

assets_path = dhm_cfg.delta_assets_path
# Files:
training_set = Path(assets_path, 'trainingsets', '2D', 'training', 'tracking_set')
validation_set = Path(assets_path, 'trainingsets', '2D', 'validation', 'tracking_set')
savefile = Path(assets_path, 'models', 'unet_pads_track.hdf5')


# Training parameters:
batch_size = 2
epochs = 300
steps_per_epoch = 1000
validation_steps = 300
patience = 50

# Data generator parameters:
train_data_gen_args = dict(
    rotation=1,
    zoom=0.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
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

# Generator init:
myGene_val = trainGenerator_track(
    batch_size,
    os.path.join(validation_set, "img"),
    os.path.join(validation_set, "seg"),
    os.path.join(validation_set, "previmg"),
    os.path.join(validation_set, "segall"),
    os.path.join(validation_set, "mot_dau"),
    os.path.join(validation_set, "wei"),
    target_size=delta_cfg.target_size_track,
    crop_windows=delta_cfg.crop_windows,
)

# Define model:
model = unet_track(pretrained_weights=savefile, input_size=delta_cfg.target_size_track + (4,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="val_loss", verbose=1, save_best_only=True
)
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)

# Train:
history = model.fit(
    myGene_train,
    validation_data=myGene_val,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping],
)
