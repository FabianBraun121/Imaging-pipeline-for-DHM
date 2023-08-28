# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:00:07 2023

@author: SWW-Bc20
"""
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\tests\\delta\\train_segmentation\\noise_0.06_blur_2_ed_sigma_50.pkl'
with open(path, 'rb') as f:
    history = pickle.load(f)



# Define custom colors
dark_blue = '#1f77b4'  # Standard blue color
light_blue = '#729ece'  # A slightly lighter shade of blue
dark_green = '#006400'
light_green = '#00cc00'

# Create figure and primary axes for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss and validation loss on primary axis
ax1.plot(history['loss'], label='Training Loss', color=dark_blue)
ax1.plot(history['val_loss'], label='Validation Loss', color=dark_green)
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Weighted binary crossentropy', fontsize=16)
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.set_yscale('log')  # Using a logarithmic scale for better visualization of loss values
ax1.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)  # Adding a dashed grid for loss

# Set y-axis boundaries for loss
ax1.set_ylim(bottom=50, top=500)  # Adjust the values as needed

# Create a secondary y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(history['unstack_acc'], label='Training Accuracy', color=light_blue)
ax2.plot(history['val_unstack_acc'], label='Validation Accuracy', color=light_green)
ax2.set_ylabel('Pixel accuracy', fontsize=16)
# ax2.tick_params(axis='y', labelcolor='green')

# Set y-axis boundaries for accuracy
ax2.set_ylim(bottom=0.99, top=1)  # Adjust the values as needed

# Adding legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

#%%
corr = np.corrcoef(np.stack((history['loss'][100:], history['val_loss'][100:])))
