# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:37 2023

@author: SWW-Bc20
"""

import matplotlib.pyplot as plt
import os
import tifffile

class ImageSliderApp:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_list = sorted(os.listdir(image_folder))
        self.current_image_index = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Image Slider")
        self.ax.axis("off")

        # Create previous and next buttons
        self.prev_button_ax = plt.axes([0.4, 0.01, 0.1, 0.05])
        self.prev_button = plt.Button(self.prev_button_ax, "Previous")
        self.prev_button.on_clicked(self.load_previous_image)

        self.next_button_ax = plt.axes([0.5, 0.01, 0.1, 0.05])
        self.next_button = plt.Button(self.next_button_ax, "Next")
        self.next_button.on_clicked(self.load_next_image)

        # Create image slider
        self.slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
        self.slider = plt.Slider(self.slider_ax, "", 0, len(self.image_list) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.load_slider_image)

        # Load the first image
        self.load_image()

        # Show the plot
        plt.show()

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
        image = tifffile.imread(image_path)
        self.ax.imshow(image)

    def load_previous_image(self, event):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
            self.slider.set_val(self.current_image_index)

    def load_next_image(self, event):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image()
            self.slider.set_val(self.current_image_index)

    def load_slider_image(self, index):
        self.current_image_index = int(index)
        self.load_image()

# Usage example
image_folder =  r'F:\Ilastik\F3_20230406\2023-04-06 11-07-24 phase averages\00001'
app = ImageSliderApp(image_folder)