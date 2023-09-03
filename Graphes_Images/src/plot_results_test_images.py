# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:51:18 2023

@author: SWW-Bc20
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.morphology import binary_dilation
from skimage.registration import phase_cross_correlation
import json
import cv2

base_dir = r'F:\test_images'

def plot_images_and_difference(image1, image2):
    # Calculate the difference between the second and first image
    image_diff = image2 - image1

    # Create a figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Set a common colorbar for all images
    vmin = min(image1.min(), image2.min(), image_diff.min())
    vmax = max(image1.max(), image2.max(), image_diff.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    lw = 2
    # Plot the first image
    im1 = axes[0].imshow(image1, cmap=cmap, norm=norm)
    axes[0].spines['top'].set_linewidth(lw)
    axes[0].spines['bottom'].set_linewidth(lw)
    axes[0].spines['left'].set_linewidth(lw)
    axes[0].spines['right'].set_linewidth(lw)
    
    # Plot the second image
    im2 = axes[1].imshow(image2, cmap=cmap, norm=norm)
    axes[1].spines['top'].set_linewidth(lw)
    axes[1].spines['bottom'].set_linewidth(lw)
    axes[1].spines['left'].set_linewidth(lw)
    axes[1].spines['right'].set_linewidth(lw)

    # Plot the difference image
    im3 = axes[2].imshow(image_diff, cmap=cmap, norm=norm)
    axes[2].spines['top'].set_linewidth(lw)
    axes[2].spines['bottom'].set_linewidth(lw)
    axes[2].spines['left'].set_linewidth(lw)
    axes[2].spines['right'].set_linewidth(lw)

    # Remove ticks and titles from all subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")

    # Add a colorbar
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = plt.colorbar(im3, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('optical path length [nm]', labelpad=-80, fontsize=25)
    cbar.ax.tick_params(labelsize=23)
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


#%%
######################## flatten numerical analysis #######################
mae_unflattend = []
mae_flattend = []
mean_bac = []
points_bac = []
data_file_dir = r'F:\test_images\data_files'
experiment_names = ['E10_20230216_2023-02-16 12-16-39', 'F3_20230406_2023-04-06 11-07-24']
for e in range(len(experiment_names)):
    for i in range(8):
        for j in range(0, 100, 25):
            kk = 0
            for k in range(25):
                filename = f'{experiment_names[e]}_{str(i+1).zfill(5)}_{str(k%5+1).zfill(5)}_{str(k//5+1).zfill(5)}_Holograms_{str(j).zfill(5)}_holo.tif'
                if not os.path.exists(base_dir + os.sep + 'without_background' + os.sep + filename):
                    continue
                
                image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename)
                mask = image_without_background>0.25
                mask = binary_dilation(binary_dilation(binary_dilation(mask)))
                
                image_unfocused_unflattened = tifffile.imread(base_dir + os.sep + 'unfocused_unflattened' + os.sep + filename)
                image_unfocused_flattened = tifffile.imread(base_dir + os.sep + 'unfocused_flattened' + os.sep + filename)
                
                uu = image_unfocused_unflattened[~mask]
                mae_unflattend.append(np.mean(np.abs(uu-np.zeros_like(uu)))*794/(2*np.pi))
                uf = image_unfocused_flattened[~mask]
                mae_flattend.append(np.mean(np.abs(uf-np.zeros_like(uf)))*794/(2*np.pi))
                
                mean_bac.append(np.mean(image_unfocused_flattened[mask])*794/(2*np.pi))
                points_bac.append(len(mask[mask==True]))
        print(f'start {i}')
# mean(mae_unflattend) = 437.5763142363454 (348.1224687131989), mean(mae_flattend) = 6.555890270638546 (0.6622838564051792)
#%%
######################## flatten image ######################
filename = 'F3_20230406_2023-04-06 11-07-24_00001_00001_00001_Holograms_00050_holo.tif'

image_uu = tifffile.imread(base_dir + os.sep + 'unfocused_unflattened' + os.sep + filename)*794/(2*np.pi)
image_uf = tifffile.imread(base_dir + os.sep + 'unfocused_flattened' + os.sep + filename)*794/(2*np.pi)
plot_images_and_difference(image_uu, image_uf)

#%%
######################## focus point numerical analysis #######################
focus_offsets = []
difs_bacteria = []
mea_difs_pixels = []
mean_bac = []
points_bac = []
data_file_dir = r'F:\test_images\data_files'
experiment_names = ['E10_20230216_2023-02-16 12-16-39', 'F3_20230406_2023-04-06 11-07-24']
data_file_names = ['data file 2023-08-22 20-16-57.json', 'data file 2023-08-23 13-12-24.json']
for e in range(len(experiment_names)):
    data_file_path = data_file_dir + os.sep + data_file_names[e] 
    with open(data_file_path, 'r') as file:
        data_file = json.load(file)
    foci = [data_file['images'][d]['foci'] for d in list(data_file['images'])]
    for i in range(8):
        for j in range(0, 100, 25):
            kk = 0
            for k in range(25):
                filename = f'{experiment_names[e]}_{str(i+1).zfill(5)}_{str(k%5+1).zfill(5)}_{str(k//5+1).zfill(5)}_Holograms_{str(j).zfill(5)}_holo.tif'
                if not os.path.exists(base_dir + os.sep + 'without_background' + os.sep + filename):
                    continue
                focus_offset = foci[i*4+j//25][kk] + 2.3
                focus_offsets.append(focus_offset)
                kk +=1
                
                image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename)
                mask = image_without_background>0.25
                mask = binary_dilation(binary_dilation(binary_dilation(mask)))
                
                ######################### unfocused vs focused #############################
                image_unfocused_flattened = tifffile.imread(base_dir + os.sep + 'unfocused_flattened' + os.sep + filename)
                image_focused_flattened = tifffile.imread(base_dir + os.sep + 'focused_flattened' + os.sep + filename)
                dif = (image_focused_flattened-image_unfocused_flattened)[mask]*794/(2*np.pi)
                points_bac.append(len(dif))
                mea_difs_pixels.append(np.mean(np.abs(dif)))
                difs_bacteria.append(np.mean(dif))
                mean_bac.append(np.mean(image_focused_flattened[mask])*794/(2*np.pi))
        print(f'position {i} done')
# np.mean(np.abs(focus_offsets)) = 0.06565876676394965 (0.06526685939550575), np.mean(np.abs(difs_bacteria)) = 0.3241210041515854 ( 0.782255995976346)
# np.mean(mea_difs_pixels) = 1.9647389743949002 (1.7299677011264283), np.mean(mean_bac) = 26.92346201646424 (3.354683722592203)
#%%
######################## focus point image ######################
# Extreme example of focus offset
filename = 'F3_20230406_2023-04-06 11-07-24_00001_00001_00001_Holograms_00050_holo.tif'

image_uf = tifffile.imread(base_dir + os.sep + 'unfocused_flattened' + os.sep + filename)*794/(2*np.pi)
image_ff = tifffile.imread(base_dir + os.sep + 'focused_flattened' + os.sep + filename)*794/(2*np.pi)
plot_images_and_difference(image_uf, image_ff)

#%%
mask = image_ff>0.25*794/(2*np.pi)
mask = binary_dilation(binary_dilation(binary_dilation(mask)))
dif = (image_ff-image_uf)[mask]
print(np.mean(np.abs(dif)))
#%%
######################## background numerical analysis #######################
mae_ff = []
mae_wb = []
mean_bac = []
points_bac = []
data_file_dir = r'F:\test_images\data_files'
experiment_names = ['E10_20230216_2023-02-16 12-16-39', 'F3_20230406_2023-04-06 11-07-24']
for e in range(len(experiment_names)):
    for i in range(8):
        for j in range(0, 100, 25):
            kk = 0
            for k in range(25):
                filename = f'{experiment_names[e]}_{str(i+1).zfill(5)}_{str(k%5+1).zfill(5)}_{str(k//5+1).zfill(5)}_Holograms_{str(j).zfill(5)}_holo.tif'
                if not os.path.exists(base_dir + os.sep + 'without_background' + os.sep + filename):
                    continue
                
                image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename)
                mask = image_without_background>0.25
                mask = binary_dilation(binary_dilation(binary_dilation(mask)))
                wb = image_without_background[~mask]
                
                image_ff = tifffile.imread(base_dir + os.sep + 'focused_flattened' + os.sep + filename)
                ff = image_ff[~mask]
                
                mae_ff.append(np.mean(np.abs(ff-np.zeros_like(ff)))*794/(2*np.pi))
                mae_wb.append(np.mean(np.abs(wb-np.zeros_like(wb)))*794/(2*np.pi))
                
                mean_bac.append(np.mean(image_without_background[mask])*794/(2*np.pi))
                points_bac.append(len(mask[mask==True]))
# np.mean(mae_ff) = 6.555953083727898 (0.6619066701507784), np.mean(mae_wb) = 4.890570396629716 (0.7166991405623114)

#%%
######################## background image ######################
filename = 'F3_20230406_2023-04-06 11-07-24_00001_00001_00001_Holograms_00050_holo.tif'

image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename)*794/(2*np.pi)
image_ff = tifffile.imread(base_dir + os.sep + 'focused_flattened' + os.sep + filename)*794/(2*np.pi)
plot_images_and_difference(image_ff, image_without_background)

#%%
######################## averaging numerical analysis #######################
mae_wb_a = []
mae_a = []
mean_bac = []
points_bac = []
data_file_dir = r'F:\test_images\data_files'
experiment_names = ['E10_20230216_2023-02-16 12-16-39', 'F3_20230406_2023-04-06 11-07-24']
data_file_names = ['data file 2023-08-22 20-16-57.json', 'data file 2023-08-23 13-12-24.json']
for e in range(len(experiment_names)):
    data_file_path = data_file_dir + os.sep + data_file_names[e] 
    with open(data_file_path, 'r') as file:
        data_file = json.load(file)
    foci = [data_file['images'][d]['foci'] for d in list(data_file['images'])]
    for i in range(8):
        for j in range(0, 100, 25):
            filename_unaveraged = f'{experiment_names[e]}_{str(i+1).zfill(5)}_{str(1).zfill(5)}_{str(1).zfill(5)}_Holograms_{str(j).zfill(5)}_holo.tif'
            filename_averaged = f'{experiment_names[e]}_{str(i+1).zfill(5)}_{str(5).zfill(5)}_{str(5).zfill(5)}_Holograms_{str(j).zfill(5)}_holo.tif'
            if not os.path.exists(base_dir + os.sep + 'without_background' + os.sep + filename_unaveraged):
                continue
            if not os.path.exists(base_dir + os.sep + 'averaged' + os.sep + filename_averaged):
                continue
            
            image_averaged = tifffile.imread(base_dir + os.sep + 'averaged' + os.sep + filename_averaged)
            mask = image_averaged>0.25
            mask = binary_dilation(binary_dilation(binary_dilation(mask)))
            avg = image_averaged[~mask]
            
            image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename_unaveraged)
            shift_measured, error, diffphase = phase_cross_correlation(image_without_background[:700,:700], image_averaged)
            if shift_measured[1]>100 or shift_measured[0]<0:
                continue
            image_without_background = image_without_background[int(shift_measured[0]):int(shift_measured[0])+700, int(shift_measured[1]):int(shift_measured[1])+700]
            wb = image_without_background[~mask]
            
            mae_wb_a.append(np.mean(np.abs(wb-np.zeros_like(wb)))*794/(2*np.pi))
            mae_a.append(np.mean(np.abs(avg-np.zeros_like(avg)))*794/(2*np.pi))
            
            mean_bac.append(np.mean(image_without_background[mask])*794/(2*np.pi))
            points_bac.append(len(mask[mask==True]))
        print(f'position {i} done')
# np.mean(mae_wb_a) = 5.062496793793493 (0.5678454313757011), np.mean(mae_a) = 3.441808529258636 (0.5794835287190342)
#%%
######################## background image ######################
filename_unaveraged = 'F3_20230406_2023-04-06 11-07-24_00001_00001_00001_Holograms_00050_holo.tif'
filename_averaged = 'F3_20230406_2023-04-06 11-07-24_00001_00005_00005_Holograms_00050_holo.tif'

image_without_background = tifffile.imread(base_dir + os.sep + 'without_background' + os.sep + filename_unaveraged)*794/(2*np.pi)
image_averaged = tifffile.imread(base_dir + os.sep + 'averaged' + os.sep + filename_averaged)*794/(2*np.pi)
shift_measured, error, diffphase = phase_cross_correlation(image_without_background[:700,:700], image_averaged)
image_without_background = image_without_background[int(shift_measured[0]):int(shift_measured[0])+700, int(shift_measured[1]):int(shift_measured[1])+700]
plot_images_and_difference(image_without_background, image_averaged)


#%%

import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Unprocessed', 'Plane\n subtracted', 'Focused', 'Background\n subtracted', 'Averaged']
mae_values = [439.5410532107403, 8.520692058122798, 6.555953083727898, 4.890570396629716, 3.441808529258636]
std_values = [348.12676715705027, 1.852271223911539, 0.6619066701507784, 0.7166991405623114, 0.5794835287190342]

# Create bar chart
x_pos = np.arange(len(labels))
plt.bar(x_pos, mae_values, yerr=std_values, align='center', alpha=0.7, capsize=10, color='blue')

# Set y-axis to be logarithmic
plt.yscale('log')

plt.yticks([1, 10, 100, 1000], fontsize=14)
# Add labels and title
plt.xticks(x_pos, labels, rotation=10, fontsize=14)
plt.ylabel('Mean absolute error (nm)', fontsize=14)
# plt.title('Decrease in Noise Level Across Processing Steps')

# Display the plot
plt.tight_layout()
plt.show()
