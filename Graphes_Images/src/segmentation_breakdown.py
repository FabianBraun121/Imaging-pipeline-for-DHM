# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:58:19 2023

@author: SWW-Bc20
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your .mp4 video file
video_path = r'F:\C11_20230217\2023-02-17 11-13-34 phase averages\00001\delta_results\Position00001.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize an empty list to store frames
frames = []

# Read frames from the video and store them in the list
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to RGB (OpenCV reads in BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

# Release the video capture object
cap.release()

# Convert the list of frames to a numpy array
video_np = np.array(frames)

images_of_interest1 = np.delete(video_np[36:43, 100:180, 150:230], [4,5], axis=0)
images_of_interest2 = np.delete(video_np[50:57, 70:130, 410:470], [1,2], axis=0)

def plot_images_side_by_side(image_series1, image_series2):
    # Create a figure with two rows and five columns
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    # Plot the first set of images in the top row
    for i in range(5):
        axes[0, i].imshow(image_series1[i])
        axes[0, i].axis('off')

    # Plot the second set of images in the bottom row
    for i in range(5):
        axes[1, i].imshow(image_series2[i])
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    
plot_images_side_by_side(images_of_interest1, images_of_interest2)