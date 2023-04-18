# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:13:32 2023

@author: SWW-Bc20
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def interactive_image_player(frames, titles=None):
    if len(frames.shape) == 3:
        num_frames, height, width = frames.shape
    else:
        num_frames, height, width, channels = frames.shape

    # Create the figure and axes objects
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)

    # Display the first frame
    im = ax.imshow(frames[0])

    # Create the slider axes object
    slider_ax = plt.axes([0.1, 0.15, 0.8, 0.05])
    slider = Slider(slider_ax, 'Frame', 0, num_frames-1, valinit=0)

    # Create the play/pause button axes object
    play_pause_ax = plt.axes([0.45, 0.025, 0.1, 0.1])
    play_pause_button = Button(play_pause_ax, label='▶', color='lightgoldenrodyellow', hovercolor='0.975')

    # Define the update function for the slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(frames[frame_idx])
        if titles is not None:
            title = titles[frame_idx]
        else:
            title = f'Frame {frame_idx}'
        ax.set_title(title, fontsize=16)
        fig.canvas.draw_idle()

    # Connect the slider update function to the slider object
    slider.on_changed(update)

    # Define the play/pause function
    def play_pause(event):
        nonlocal playing
        playing = not playing
        if playing:
            play_pause_button.label.set_text('❚❚')
            for i in range(int(slider.val), num_frames):
                slider.set_val(i)
                plt.pause(0.01)
                if not playing:
                    break
            if playing:
                play_pause_button.label.set_text('▶')
        else:
            play_pause_button.label.set_text('▶')
    
    # Define the key press function
    def key_press(event):
        nonlocal playing
        if event.key == ' ':
            play_pause(None)
        elif event.key == 'right':
            if slider.val < num_frames-1:
                slider.set_val(slider.val + 1)
        elif event.key == 'left':
            if slider.val > 0:
                slider.set_val(slider.val - 1)

    # Initialize the playing flag to False
    playing = False

    # Connect the play/pause function to the play/pause button object
    play_pause_button.on_clicked(play_pause)
    
    # Connect the key press function to the figure object
    fig.canvas.mpl_connect('key_press_event', key_press)

    # Set the title of the first frame
    if titles is not None:
        title = titles[0]
    else:
        title = 'Frame 0'
    ax.set_title(title, fontsize=16)
    
    # Show the plot
    plt.show()