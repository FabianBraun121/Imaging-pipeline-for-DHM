# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:46:41 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from config import Config
import tifffile
import numpy as np
import skimage.transform as trans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pygetwindow as gw

sys.path.append("..")
from src.spatial_averaging.utilities import PolynomialPlaneSubtractor, Koala

class ConfigEditorGUI:
    def __init__(self, root, config):
        self.root = root
        self.root.title("Config Editor")
        self.cfg = config

        # Create and set up the notebook (tabbed layout)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Frame for OK button
        self.ok_button_frame = ttk.Frame(root)
        ok_button = ttk.Button(self.ok_button_frame, text="OK", command=self.close_editor)
        ok_button.pack(pady=10)
        self.ok_button_frame.pack(side="bottom", fill="x")


        self.general_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.general_tab, text="General")
        self.create_general_tab()
        
        self.reconstruction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reconstruction_tab, text="Reconstruction")
        self.create_reconstruction_tab()
        
        self.create_image_type_tab('BF')
        self.create_image_type_tab('PC')
        
        self.delta_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.delta_tab, text="Delta")
        self.create_delta_tab()
        
        
    def close_editor(self):
        if not os.path.isdir(self.base_dir_var.get()):
            tk.messagebox.showerror("Error", "A valid base Directory is needed")
        elif not self.koala_config_nr_var.get().isdigit():
            tk.messagebox.showerror("Error", "A valid Koala Configuration Number is needed")
        else:
            self.root.destroy()
    
    def bind_events(self, widget, events, handler):
        for event in events:
            widget.bind(event, handler)
            
    
    
    ###################################### General TAB ######################################
    def create_general_tab(self):
        self.create_path_frame(self.general_tab, 'base_dir', "Base Directory", self.update_base_dir).pack()
        self.create_path_frame(self.general_tab, 'saving_dir', "Saving Directory", self.update_saving_dir).pack()
        pos_time_frame = ttk.Frame(self.general_tab)
        pos_frame = self.create_cb_min_max_frame(pos_time_frame, 'restrict_positions', "Select Positions")
        time_frame = self.create_cb_min_max_frame(pos_time_frame, 'restrict_timesteps', "Select Timesteps")
        pos_frame.pack(side='left', padx=(0,40))
        time_frame.pack(side='right')
        pos_time_frame.pack(pady=20)
        koala_save_frame = ttk.Frame(self.general_tab)
        self.create_koala_nr_frame(koala_save_frame).pack(side='left', padx=20)
        self.create_save_options_frame(koala_save_frame).pack(side='left', padx=20)
        koala_save_frame.pack()
    
    
    ######### General TAB: directory frame #########
    def create_path_frame(self, tab, config_variable, title_text, update_function):
        frame = ttk.Frame(tab)
        var = tk.StringVar(value=str(self.cfg.get_config_setting(config_variable)))
        setattr(self, f"{config_variable}_var", var)
        entry = tk.Entry(frame, textvariable=var, width=80)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event, func=update_function: func())
        button = tk.Button(frame, text="Select Folder", command=lambda: self.select_general_folder(config_variable))
        tk.Label(frame, text=title_text).pack()
        entry.pack(side='left')
        button.pack(side='right')
        return frame
    
    def select_general_folder(self, config_variable):
        selected_folder = filedialog.askdirectory()
        if selected_folder:
            normalized_path = os.path.normpath(selected_folder)
            var = getattr(self, f"{config_variable}_var")
            var.set(normalized_path)
            self.cfg.set_config_setting(config_variable, normalized_path)
            if config_variable == 'base_dir':
                self.update_saving_dir_from_base_dir()
    
    def update_base_dir(self):
        path = os.path.normpath(self.base_dir_var.get())
        if os.path.isdir(path):
            self.base_dir_var.set(path)
            self.cfg.set_config_setting('base_dir', path)
            self.update_saving_dir_from_base_dir()
        else: 
            # A pop up should be added "Directory does not exist"
            tk.messagebox.showerror("Error", "Directory does not exist")
            self.base_dir_var.set(self.cfg.get_config_setting('base_dir'))
    
    def update_saving_dir_from_base_dir(self):
        if self.cfg.get_config_setting('saving_dir') == None:
            saving_dir = self.cfg.get_config_setting('base_dir') + " processed"
            self.saving_dir_var.set(saving_dir)
            self.cfg.set_config_setting('saving_dir', saving_dir)
    
    def update_saving_dir(self):
        path = os.path.normpath(self.saving_dir_var.get())
        self.saving_dir_var.set(path)
        self.cfg.set_config_setting('saving_dir', path)
    
    
    ######### General TAB: select position/timestep frame #########
    def create_cb_min_max_frame(self, tab, config_variable, title_text):
        frame = ttk.Frame(tab)
        all_var = tk.BooleanVar(value=self.cfg.get_config_setting(config_variable)==None)
        min_var = tk.StringVar()
        setattr(self, f"{config_variable}_min_var", min_var)
        max_var = tk.StringVar()
        setattr(self, f"{config_variable}_max_var", max_var)
        min_entry = tk.Entry(frame, textvariable=min_var, width=10)
        self.bind_events(min_entry, ("<FocusOut>", "<Return>"), lambda event, func=self.update_min_max: func(config_variable))
        max_entry = tk.Entry(frame, textvariable=max_var, width=10)
        self.bind_events(max_entry, ("<FocusOut>", "<Return>"), lambda event, func=self.update_min_max: func(config_variable))
        self.toggle_min_max(all_var, min_entry, max_entry, config_variable)
        all_checkbutton = tk.Checkbutton(frame, variable=all_var, command=lambda: self.toggle_min_max(all_var, min_entry, max_entry, config_variable))
        tk.Label(frame, text=title_text).pack()
        tk.Label(frame, text='All').pack(side='left')
        all_checkbutton.pack(side='left', padx=(0,5))
        tk.Label(frame, text='Min').pack(side='left')
        min_entry.pack(side='left', padx=(0,5))
        tk.Label(frame, text='Max').pack(side='left')
        max_entry.pack(side='left')
        return frame
    
    def toggle_min_max(self, all_var, min_entry, max_entry, config_variable):
        # Toggle the state of entries based on the "all" Checkbutton state
        state = tk.DISABLED if all_var.get() else tk.NORMAL
        min_entry["state"] = state
        max_entry["state"] = state
        getattr(self, f"{config_variable}_min_var").set(0)
        getattr(self, f"{config_variable}_max_var").set(-1)
        if all_var.get():
            self.cfg.set_config_setting(config_variable, None)
        else:
            self.cfg.set_config_setting(config_variable, (0,-1))
            
            
    def update_min_max(self, config_variable):
        min_value = int(getattr(self, f"{config_variable}_min_var").get())
        max_value = int(getattr(self, f"{config_variable}_max_var").get())
        getattr(self, f"{config_variable}_min_var").set(min_value)
        getattr(self, f"{config_variable}_max_var").set(max_value)
        self.cfg.set_config_setting(config_variable, (min_value, max_value))
    
    
    ######### General TAB: select save option frame #########
    def create_save_options_frame(self, tab):
        frame = ttk.Frame(tab)
        save_as_bulk_var = tk.BooleanVar(value=self.cfg.get_config_setting('save_as_bulk'))
        setattr(self, "save_as_bulk_var", save_as_bulk_var)
        save_as_bulk_checkbutton = tk.Checkbutton(frame, variable=save_as_bulk_var, command=lambda: self.cfg.set_config_setting('save_as_bulk', self.save_as_bulk_var.get()))
        save_format_var = tk.StringVar(value=self.cfg.get_config_setting('save_format'))
        setattr(self, "save_format_var", save_format_var)
        save_format_combobox = ttk.Combobox(frame, textvariable=save_format_var, values=[".bin", ".tif"], width=10)
        save_format_combobox.bind("<<ComboboxSelected>>", lambda event: self.save_format_on_combobox_selected())
        tk.Label(frame, text="Saving Options").pack()
        tk.Label(frame, text='save as bulk').pack(side='left')
        save_as_bulk_checkbutton.pack(side='left',padx=(5,15))
        tk.Label(frame, text='format').pack(side='left', padx=5)
        save_format_combobox.pack()
        return frame
    
    def save_format_on_combobox_selected(self):
        # Your code to handle the selection
        selected_value = self.save_format_var.get()
        self.cfg.set_config_setting('save_format', selected_value)
    
        # Set focus to another widget (e.g., the root window)
        self.root.focus_set()
                
    
    ######### General TAB: Koala config nr frame #########
    def create_koala_nr_frame(self, tab):
        frame = ttk.Frame(tab)
        koala_config_nr_var = tk.StringVar(value=self.cfg.get_config_setting('koala_config_nr'))
        setattr(self, "koala_config_nr_var", koala_config_nr_var)
        koala_config_nr_entry = tk.Entry(frame, textvariable=koala_config_nr_var, width=10)
        self.bind_events(koala_config_nr_entry, ("<FocusOut>", "<Return>"), lambda event: self.update_konfig_nr())
        tk.Label(frame, text='Koala').pack()
        tk.Label(frame, text='Config Nr.').pack(side='left', padx=5)
        koala_config_nr_entry.pack(side='left', padx=5)
        return frame
    
    def update_konfig_nr(self):
        try: 
            koala_config_nr = int(getattr(self, "koala_config_nr_var").get())
            getattr(self, "koala_config_nr_var").set(koala_config_nr)
            self.cfg.set_config_setting('koala_config_nr', koala_config_nr)
        except ValueError:
            tk.messagebox.showerror("Error", "Needs to be an integer number")
    
    
    
    ###################################### Reconstruction TAB ######################################
    def create_reconstruction_tab(self):
        self.create_recon_settings_frame(self.reconstruction_tab).pack(pady=10)
        self.create_recon_size_frame(self.reconstruction_tab).pack(pady=10)
        self.create_focus_search_frame(self.reconstruction_tab).pack(pady=10)

    
    ######### Reconstruction TAB: recon frame #########
    def create_recon_settings_frame(self, tab):
        frame = ttk.Frame(tab)
        tk.Label(frame, text='Reconstruction settings').pack()
        self.create_number_entry(frame, 'reconstruction_distance_low', 'Dist. low', self.update_float_entry)
        self.create_number_entry(frame, 'reconstruction_distance_high', 'Dist. high', self.update_float_entry)
        self.create_number_entry(frame, 'plane_fit_order', 'Sub. plane degree', self.update_int_entry)
        self.create_number_entry(frame, 'nfev_max', 'Max. func. eval.', self.update_int_entry)
        return frame
    
    def create_number_entry(self, frame, config_name, label_text, update_function):
        var = tk.StringVar(value=self.cfg.get_config_setting(config_name))
        setattr(self, f"{config_name}_var", var)
        entry = tk.Entry(frame, textvariable=var, width=7)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: update_function(config_name))
        tk.Label(frame, text=label_text).pack(side='left', padx=(15,5))
        entry.pack(side='left')
    
    def update_float_entry(self, name):
        try:
            dist = float(getattr(self, f"{name}_var").get())
            getattr(self, f"{name}_var").set(dist)
            self.cfg.set_config_setting(name, dist)
        except ValueError:
            tk.messagebox.showerror("Error", "Needs to be an number")
    
    def update_int_entry(self, name):
        try:
            dist = int(getattr(self, f"{name}_var").get())
            getattr(self, f"{name}_var").set(dist)
            self.cfg.set_config_setting(name, dist)
        except ValueError:
            tk.messagebox.showerror("Error", "Needs to be an number")
            
    ######### Reconstruction TAB: recon frame #########
    def create_recon_size_frame(self, tab):
        frame = ttk.Frame(tab)        
        
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            var = tk.StringVar(value='0')
            setattr(self, f"recon_{corner}_var", var)
        
            entry = tk.Entry(frame, text=corner, textvariable=var, width=10)
            entry['state'] = tk.DISABLED
            setattr(self, f"recon_{corner}_entry", entry)
            self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event, func=self.update_recon_corners: func())
        
        recon_var = tk.BooleanVar(value=False)
        setattr(self, "recon_var", recon_var)
        recon_checkbutton = tk.Checkbutton(frame, variable=recon_var, command=lambda: self.toggle_recon())
        
        recon_all_the_same_var = tk.BooleanVar(value=False)
        setattr(self, "recon_all_the_same_var", recon_all_the_same_var)
        recon_all_the_same_checkbutton = tk.Checkbutton(frame, variable=recon_all_the_same_var, command=lambda: self.toggle_recon_corners())
        setattr(self, "recon_all_the_same_checkbutton", recon_all_the_same_checkbutton)
        
        recon_select_on_image_var = tk.BooleanVar(value=False)
        setattr(self, "recon_select_on_image_var", recon_select_on_image_var)
        recon_select_on_image_checkbutton = tk.Checkbutton(frame, variable=recon_select_on_image_var, command=lambda: self.toggle_recon_corners())
        setattr(self, "recon_select_on_image_checkbutton", recon_select_on_image_checkbutton)
        
        self.toggle_recon()
        tk.Label(frame, text='Reconstruction rectangle').pack()
        tk.Label(frame, text='Change').pack(side='left')
        recon_checkbutton.pack(side='left', padx=(0,5))
        tk.Label(frame, text='Same').pack(side='left')
        recon_all_the_same_checkbutton.pack(side='left', padx=(0,5))
        tk.Label(frame, text='on Image').pack(side='left')
        recon_select_on_image_checkbutton.pack(side='left')
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            tk.Label(frame, text=corner).pack(side='left')
            getattr(self, f"recon_{corner}_entry").pack(side='left')
        return frame
    
    def toggle_recon(self):
        if getattr(self, "recon_var").get():
            state = tk.NORMAL
        else:
            state = tk.DISABLED
            self.cfg.set_config_setting('recon_corners', None)
        getattr(self, "recon_all_the_same_checkbutton")["state"] = state
        getattr(self, "recon_select_on_image_checkbutton")["state"] = state
        self.toggle_recon_corners()
        self.cfg.set_config_setting('recon_rectangle', getattr(self, "recon_var").get())
    
    def toggle_recon_corners(self):
        self.cfg.set_config_setting('recon_select_on_image', getattr(self, "recon_select_on_image_var").get())
        self.cfg.set_config_setting('recon_all_the_same', getattr(self, "recon_all_the_same_var").get())
        if not getattr(self, "recon_select_on_image_var").get() and getattr(self, "recon_all_the_same_var").get() and getattr(self, "recon_var").get():
            state = tk.NORMAL
            self.cfg.set_config_setting('recon_corners', ((0,0),(0,0)))
        else:
            state = tk.DISABLED
            self.cfg.set_config_setting('recon_corners', None)
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            getattr(self, f"recon_{corner}_var").set(0)
            getattr(self, f"recon_{corner}_entry")['state'] = state
        
    def update_recon_corners(self):
        xmin = int(self.recon_xmin_var.get())
        xmax = int(self.recon_xmax_var.get())
        ymin = int(self.recon_ymin_var.get())
        ymax = int(self.recon_ymax_var.get())
        self.cfg.set_config_setting('recon_corners', ((ymin,ymax),(xmin,xmax)))
        
    ######### Reconstruction TAB: focus search #########
    def create_focus_search_frame(self, tab):
        frame = ttk.Frame(tab)
        nfevals = self.cfg.get_config_setting('nfevaluations')
        methods = self.cfg.get_config_setting('focus_method')
        self.focus_searches = len(nfevals)
        for i in range(len(nfevals)):
            self.create_focus_search_entry(frame, i, nfevals[i], methods[i])
        self.create_focus_search_entry(frame, len(nfevals), '', 'std_amp')
        return frame
    
    def create_focus_search_entry(self, frame, number, nfev, method):
        focus_search_frame = ttk.Frame(frame)
        var = tk.StringVar(value=nfev)
        setattr(self, f'nfevaluations_{number}_var', var)
        entry = tk.Entry(focus_search_frame, textvariable=var, width=10)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_focus_search_entries(frame))
        var = tk.StringVar(value=method)
        setattr(self, f'focus_method_{number}_var', var)
        combobox = ttk.Combobox(focus_search_frame, textvariable=var, values=['std_amp', 'phase_sharpness', 'combined'], width=20)
        combobox.bind("<<ComboboxSelected>>", lambda event: self.update_focus_search_methods())
        tk.Label(focus_search_frame, text=f'Search {number+1}: Num. Func. Evaluations ').pack(side='left', padx=5)
        entry.pack(side='left', padx=5)
        tk.Label(focus_search_frame, text='Method').pack(side='left', padx=5)
        combobox.pack(side='left', padx=5)
        focus_search_frame.pack()
        setattr(self, f"focus_search_frame_{number}", focus_search_frame)
    
    def update_focus_search_entries(self, frame):
        number = self.focus_searches
        nfevaluations = []
        for i in range(number):
            try:
                eval_number = int(getattr(self, f"nfevaluations_{i}_var").get())
                getattr(self, f"nfevaluations_{i}_var").set(eval_number)
                nfevaluations.append(eval_number)
            except ValueError:
                if i == number-1 and not getattr(self, f"nfevaluations_{number}_var").get().isdigit():
                    getattr(self, f"nfevaluations_{i}_var").set('')
                    self.focus_searches -= 1
                    getattr(self, f"focus_search_frame_{i+1}").pack_forget()
                else:
                    tk.messagebox.showerror("Error", "Needs to be an number")
        try:
            eval_number = int(getattr(self, f"nfevaluations_{number}_var").get())
            getattr(self, f"nfevaluations_{number}_var").set(eval_number)
            nfevaluations.append(eval_number)
            self.create_focus_search_entry(frame, number+1, '', 'std_amp')
            self.focus_searches += 1
        except ValueError:
            getattr(self, f"nfevaluations_{number}_var").set('')
        
        self.cfg.set_config_setting('nfevaluations', tuple(nfevaluations))
    
    def update_focus_search_methods(self):
        number = self.focus_searches
        methods = []
        for i in range(number):
            method = getattr(self, f"focus_method_{i}_var").get()
            methods.append(method)
        self.cfg.set_config_setting('focus_method', tuple(methods))


    ###################################### IMAGE TYPE TAB ######################################
    def create_image_type_tab(self, image_type):
        frame = ttk.Frame(self.notebook)
        setattr(self, f'{image_type.lower()}_tab', frame)
        self.notebook.add(getattr(self, f'{image_type.lower()}_tab'), text=f'{image_type} Images')
        
        cb_frame = ttk.Frame(frame)
        var = tk.BooleanVar(value=image_type in self.cfg.get_config_setting('additional_image_types'))
        setattr(self, f'{image_type.lower()}_var', var)
        checkbutton = tk.Checkbutton(cb_frame, variable=var, command=lambda: self.toggle_image_type(image_type))
        tk.Label(cb_frame, text=f'Process {image_type} Images').pack(side='left')
        checkbutton.pack(side='left')
        cb_frame.pack(pady=5)
        
        corners_frame = ttk.Frame(frame)
        for i, corner in enumerate(['ymin', 'ymax', 'xmin', 'xmax']):
            var = tk.StringVar(value=self.cfg.get_config_setting(f'{image_type.lower()}_cut')[i//2][i%2])
            setattr(self, f'{image_type.lower()}_{corner}_var', var)
        
            entry = tk.Entry(corners_frame, text=corner, textvariable=var, width=10)
            entry['state'] = tk.NORMAL if getattr(self, f'{image_type.lower()}_var').get() else tk.DISABLED
            setattr(self, f'{image_type.lower()}_{corner}_entry', entry)
            self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_image_type_corners(image_type))
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            tk.Label(corners_frame, text=corner).pack(side='left')
            getattr(self, f"{image_type.lower()}_{corner}_entry").pack(padx=(5,10), side='left')
        button = tk.Button(corners_frame, text="Select Corners on Image", command=lambda: self.select_corners_on_image(image_type))
        button.pack(padx=10)
        corners_frame.pack(pady=5)
        
        searches_frame = ttk.Frame(frame)
        var = tk.StringVar(value=self.cfg.get_config_setting(f'{image_type.lower()}_local_searches'))
        setattr(self, f'{image_type.lower()}_local_searches_var', var)
        entry = tk.Entry(searches_frame, textvariable=var, width=5)
        entry['state'] = tk.NORMAL if getattr(self, f'{image_type.lower()}_var').get() else tk.DISABLED
        setattr(self, f'{image_type.lower()}_local_searches_entry', entry)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_int_entry(f'{image_type.lower()}_local_searches'))
        tk.Label(searches_frame, text='Grid searches: ').pack(side='left')
        entry.pack(padx=5, side='left')
        searches_frame.pack(pady=5)
        
        grid_search_frame = ttk.Frame(frame)
        self.create_gride_side_frame(grid_search_frame, image_type, 'rot', 'Rotation').pack(side='left', padx=20)
        self.create_gride_side_frame(grid_search_frame, image_type, 'zoom', 'Zoom').pack(side='right', padx=20)
        grid_search_frame.pack()
        
        
    def create_gride_side_frame(self, frame, image_type, variable, text):
        frame = ttk.Frame(frame)
        tk.Label(frame, text=text).pack()
        self.create_float_entry_frame(frame, image_type, f'{image_type.lower()}_{variable}_guess', 'Guess: ').pack(side='left')
        self.create_float_entry_frame(frame, image_type, f'{image_type.lower()}_{variable}_search_length', 'Search length: ').pack(side='left')
        return frame
        
    def create_float_entry_frame(self, frame, image_type, name, text):
        frame = ttk.Frame(frame)
        var = tk.StringVar(value=self.cfg.get_config_setting(name))
        setattr(self, f'{name}_var', var)
        entry = tk.Entry(frame, textvariable=var, width=5)
        entry['state'] = tk.NORMAL if getattr(self, f'{image_type.lower()}_var').get() else tk.DISABLED
        setattr(self, f'{name}_entry', entry)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_float_entry(name))
        tk.Label(frame, text=text).pack(side='left')
        entry.pack(side='left', padx=(5,10))
        return frame
        
            
    def select_corners_on_image(self, image_type):
        base_dir = self.cfg.get_config_setting('base_dir')
        if base_dir is None:
            tk.messagebox.showerror("Error", "Select a base dir first")
            return
        if not os.path.isdir(base_dir):
            tk.messagebox.showerror("Error", "Base directory is not valid")
            return
        pos_dir = base_dir + os.sep + os.listdir(base_dir)[0]
        image_type_file_paths = [pos_dir + os.sep + f for f in os.listdir(pos_dir) if f.endswith(f'{image_type}.tif')]
        if len(image_type_file_paths) == 0:
            tk.messagebox.showerror("Error", f"No {image_type} in expected folder: {pos_dir}")
            return
        if not isinstance(self.cfg.get_config_setting('koala_config_nr'), int):
            tk.messagebox.showerror("Error", "Select a Koala configuration number")
            return
        holo_dir = pos_dir + os.sep + [f for f in os.listdir(pos_dir) if os.path.isdir(pos_dir+ os.sep + f)][0] + os.sep + 'Holograms'
        holo_path = holo_dir + os.sep + os.listdir(holo_dir)[0]
        Koala.connect(self.cfg.get_config_setting('koala_config_nr'))
        Koala.load_hologram(holo_path)
        Koala._host.SetUnwrap2DState(True)
        Koala.set_reconstruction_distance((self.cfg.get_config_setting('reconstruction_distance_low')+self.cfg.get_config_setting('reconstruction_distance_high'))/2)
        ph = Koala.get_phase_image()
        ph = PolynomialPlaneSubtractor.subtract_plane(ph, self.cfg.get_config_setting('plane_fit_order'))
        
        corners = RectangleSelector(image_type_file_paths, ph).get_corners()
        self.cfg.set_config_setting('{image_type.lower()}_cut', corners)
        for i, corner in enumerate(['ymin', 'ymax', 'xmin', 'xmax']):
            getattr(self, f'{image_type.lower()}_{corner}_var').set(corners[i//2][i%2])
        gw.getWindowsWithTitle("Config Editor")[0].activate()
            
    
    def toggle_image_type(self, image_type):
        if getattr(self, f'{image_type.lower()}_var').get():
            state = tk.NORMAL
            image_types = tuple(tuple(self.cfg.get_config_setting('additional_image_types')) + (image_type,))
        else:
            state = tk.DISABLED
            image_types = tuple(x for x in self.cfg.get_config_setting('additional_image_types') if x != image_type)
        self.cfg.set_config_setting('additional_image_types', image_types)
        for corner in ['ymin', 'ymax', 'xmin', 'xmax']:
            entry = getattr(self, f"{image_type.lower()}_{corner}_entry")
            entry['state'] = state
        getattr(self, f"{image_type.lower()}_local_searches_entry")['state'] = state
        for i in ['rot', 'zoom']:
            for j in ['guess', 'search_length']:
                getattr(self, f"{image_type.lower()}_{i}_{j}_entry")['state'] = state
        
    def update_image_type_corners(self, image_type):
        xmin = getattr(self, f"{image_type.lower()}_xmin_var").get()
        xmax = getattr(self, f"{image_type.lower()}_xmax_var").get()
        ymin = getattr(self, f"{image_type.lower()}_ymin_var").get()
        ymax = getattr(self, f"{image_type.lower()}_ymax_var").get()
        self.cfg.set_config_setting('{image_type.lower()}_cut', ((ymin,ymax),(xmin,xmax)))
    
        
        
    ###################################### DELTA TAB ######################################
    def create_delta_tab(self):
        self.create_delta_core_frame(self.delta_tab).pack(pady=10)
    
    ######### Delta TAB: core frame #########
    def create_delta_core_frame(self, tab):
        frame = ttk.Frame(tab)
        core_type_frame = ttk.Frame(frame)
        
        delta_bacteria_core_var = tk.StringVar(value=self.cfg.get_config_setting('delta_bacteria_core'))
        setattr(self, 'delta_bacteria_core_var', delta_bacteria_core_var)
        combobox = ttk.Combobox(core_type_frame, textvariable=delta_bacteria_core_var, values=['PH', 'BF'], width=5)
        combobox.bind("<<ComboboxSelected>>", lambda event: self.update_bacteria_core())
        tk.Label(core_type_frame, text='Core caluclated on').pack(side='left', padx=5)
        combobox.pack(side='left', padx=5)
        core_type_frame.pack()
        
        self.create_path_input_frame(frame, 'core_seg', 'Segmentation:')
        self.create_path_input_frame(frame, 'track', 'Tracking:')
        
        tk.Label(frame, text='Full bacteria Segmentation:').pack(pady=(10,0))
        path_frame = ttk.Frame(frame)
        var = tk.StringVar(value=self.cfg.get_config_setting('model_file_ph_full_seg'))
        setattr(self, 'model_file_ph_full_seg', var)
        entry = tk.Entry(path_frame, textvariable=var, width=110)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_full_file_path())
        button = tk.Button(path_frame, text="Select Folder", command=lambda: self.select_full_file())
        entry.pack(side='left')
        button.pack(side='right')
        path_frame.pack()
        
        return frame
    
    def create_path_input_frame(self, frame, process, titletext):
        tk.Label(frame, text=titletext).pack(pady=(10,0))
        path_frame = ttk.Frame(frame)
        core_type = getattr(self, 'delta_bacteria_core_var').get().lower()
        var = tk.StringVar(value=self.cfg.get_config_setting(f'model_file_{core_type}_{process}'))
        setattr(self, f'model_file_{process}_var', var)
        entry = tk.Entry(path_frame, textvariable=var, width=110)
        self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_file_path(process))
        button = tk.Button(path_frame, text="Select Folder", command=lambda: self.select_file(process))
        entry.pack(side='left')
        button.pack(side='right')
        path_frame.pack()
        return path_frame
    
    def select_file(self, process):
        selected_file = filedialog.askopenfilename()
        if selected_file:
            normalized_path = os.path.normpath(selected_file)
            getattr(self, f'model_file_{process}_var').set(normalized_path)
            core_type = getattr(self, "delta_bacteria_core_var").get().lower()
            self.cfg.set_config_setting(f'model_file_{core_type}_{process}', normalized_path)
    
    def select_full_file(self):
        selected_file = filedialog.askopenfilename()
        if selected_file:
            normalized_path = os.path.normpath(selected_file)
            getattr(self, 'model_file_ph_full_seg_var').set(normalized_path)
            self.cfg.set_config_setting('model_file_ph_full_seg', normalized_path)
            
    def update_bacteria_core(self):
        core_type = getattr(self, "delta_bacteria_core_var").get()
        self.cfg.set_config_setting('delta_bacteria_core', core_type)
        getattr(self, "model_file_core_seg_var").set(self.cfg.get_config_setting(f'model_file_{core_type.lower()}_core_seg'))
        getattr(self, "model_file_track_var").set(self.cfg.get_config_setting(f'model_file_{core_type.lower()}_track')) 
    
    def update_file_path(self, process):
        path = os.path.normpath(getattr(self, "model_file_{process}_var").get())
        core_type = getattr(self, "delta_bacteria_core_var").get().lower()
        if os.path.isfile(path):
            getattr(self, "model_file_{process}_var").set(path)
            self.cfg.set_config_setting(f'model_file_{core_type}_{process}', path)
        else:
            tk.messagebox.showerror("Error", "File does not exist")
            getattr(self, "model_file_{process}_var").set(self.cfg.get_config_setting(f'model_file_{core_type}_{process}'))
    
    def update_full_file_path(self):
        path = os.path.normpath(getattr(self, "model_file_ph_full_seg_var").get())
        if os.path.isfile(path):
            getattr(self, "model_file_ph_full_seg_var").set(path)
            self.cfg.set_config_setting('model_file_ph_full_seg', path)
        else:
            tk.messagebox.showerror("Error", "File does not exist")
            getattr(self, "model_file_ph_full_seg_var").set(self.cfg.get_config_setting('model_file_ph_full_seg'))
        

class RectangleSelector:
    def __init__(self, image_path, ref_image):
        self.ref_image = ref_image
        self.image_path = image_path
        self.image = trans.rotate(np.fliplr(np.array(tifffile.imread(image_path))), -90, mode="edge")
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title("Select by clicking the top left corner")
        self.ax2.set_title("Reference phase image")
        self.ax1.imshow(self.image)
        self.ax2.imshow(ref_image)
        self.output_shape = (1024,1024)
        self.x = None
        self.y = None
        self.rect = None
        self.corners = None
        self.enter_pressed = False
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        self.wait_for_enter()
        
    def on_click(self, event):
        if self.rect is not None:
            self.rect.remove()
        self.x, self.y = int(event.xdata), int(event.ydata)
        self.rect = Rectangle((self.x, self.y), self.output_shape[0], self.output_shape[1], edgecolor='r', facecolor='none')
        self.ax1.add_patch(self.rect)
        self.fig.canvas.draw()
                
    def wait_for_enter(self):
        def on_key(event):
            if event.key == 'enter':
                self.enter_pressed = True

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        while not self.enter_pressed:
            plt.pause(0.1)
        if self.rect is not None:
            if 0<=self.y<self.image.shape[0]-self.output_shape[0] and 0<=self.x<self.image.shape[1]-self.output_shape[1]:
                plt.close()
                self.corners = ((self.y, self.y+self.output_shape[0]), (self.x, self.x+self.output_shape[1]))
            else:
                 self.show_popup("Error", "Invalid Rectangle")
                 self.enter_pressed = False
                 self.wait_for_enter()
        else:
            self.show_popup("Error", "Select a Rectangle by clicking on the image")
            self.enter_pressed = False
            self.wait_for_enter()
        
    def get_corners(self):
        return self.corners
    
    @staticmethod
    def show_popup(title, message):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
        
def run_gui(config):
    root = tk.Tk()
    ConfigEditorGUI(root, config)
    root.mainloop()
    saving_dir = config.get_config_setting('saving_dir')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir) 

# config = Config()
# root = tk.Tk()
# app = ConfigEditorGUI(root, config)
# root.mainloop()
















