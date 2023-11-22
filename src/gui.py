# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:46:41 2023

@author: SWW-Bc20
"""
import os
os.chdir(os.path.dirname(__file__))
import tkinter as tk
from tkinter import ttk, filedialog
from config import Config

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
        
        # self.delta_tab = ttk.Frame(self.notebook)
        # self.notebook.add(self.delta_tab, text="Delta")
        # self.create_delta_tab()
        
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
        getattr(self, f"{config_variable}_max_var").set(0)
        if all_var.get():
            self.cfg.set_config_setting(config_variable, None)
        else:
            self.cfg.set_config_setting(config_variable, (0,0))
            
            
    def update_min_max(self, config_variable):
        min_value = int(getattr(self, f"{config_variable}_min_var").get())
        max_value = int(getattr(self, f"{config_variable}_max_var").get())
        if min_value >= 0 and max_value >= 0:
            getattr(self, f"{config_variable}_min_var").set(min_value)
            getattr(self, f"{config_variable}_max_var").set(max_value)
            self.cfg.set_config_setting(config_variable, (min_value, max_value))
        else:
            tk.messagebox.showerror("Error", "Invalid Min or Max")
    
    
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
        self.create_number_entry(frame, 'reconstruction_distance_low', 'Dist. low', self.update_recon_dist)
        self.create_number_entry(frame, 'reconstruction_distance_high', 'Dist. high', self.update_recon_dist)
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
    
    def update_recon_dist(self, name):
        try:
            dist = float(getattr(self, f"{name}_var").get())
            getattr(self, f"{name}_var").set(dist)
            self.cfg.set_config_setting('name', dist)
        except ValueError:
            tk.messagebox.showerror("Error", "Needs to be an number")
    
    def update_int_entry(self, name):
        try:
            dist = int(getattr(self, f"{name}_var").get())
            getattr(self, f"{name}_var").set(dist)
            self.cfg.set_config_setting('name', dist)
        except ValueError:
            tk.messagebox.showerror("Error", "Needs to be an number")
            
    ######### Reconstruction TAB: recon frame #########
    def create_recon_size_frame(self, tab):
        frame = ttk.Frame(tab)        
        corner_vars = {}
        corner_entries = {}
        
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            var = tk.StringVar(value='0')
            setattr(self, f"recon_{corner}_var", var)
            corner_vars[corner] = var
        
            entry = tk.Entry(frame, text=corner, textvariable=var, width=10)
            entry['state'] = tk.DISABLED
            self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event, func=self.update_recon_corners: func())
            corner_entries[corner] = entry
        
        recon_var = tk.BooleanVar(value=False)
        recon_all_the_same_var = tk.BooleanVar(value=False)
        recon_select_on_image_var = tk.BooleanVar(value=False)
        recon_all_the_same_checkbutton = tk.Checkbutton(frame, variable=recon_all_the_same_var,
                                                        command=lambda: self.toggle_recon_corners(recon_select_on_image_var, recon_all_the_same_var, corner_entries,
                                                                                                  recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton))
        recon_select_on_image_checkbutton = tk.Checkbutton(frame, variable=recon_select_on_image_var,
                                                           command=lambda: self.toggle_recon_corners(recon_select_on_image_var, recon_all_the_same_var, corner_entries,
                                                                                                     recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton))
        recon_checkbutton = tk.Checkbutton(frame, variable=recon_var,
                                           command=lambda: self.toggle_recon(recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton, corner_entries))
        self.toggle_recon(recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton, corner_entries)
        tk.Label(frame, text='Reconstruction rectangle').pack()
        tk.Label(frame, text='Change').pack(side='left')
        recon_checkbutton.pack(side='left', padx=(0,5))
        tk.Label(frame, text='Same').pack(side='left')
        recon_all_the_same_checkbutton.pack(side='left', padx=(0,5))
        tk.Label(frame, text='on Image').pack(side='left')
        recon_select_on_image_checkbutton.pack(side='left')
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            tk.Label(frame, text=corner).pack(side='left')
            corner_entries[corner].pack(side='left')
        return frame
    
    def toggle_recon(self, recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton, corner_entries):
        if recon_var.get():
            state = tk.NORMAL
        else:
            state = tk.DISABLED
            for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
                corner_entries[corner]['state'] = state
        recon_all_the_same_checkbutton["state"] = state
        recon_select_on_image_checkbutton["state"] = state
        self.cfg.set_config_setting('recon_rectangle', recon_var.get())
    
    def toggle_recon_corners(self, recon_select_on_image_var, recon_all_the_same_var, corner_entries,
                             recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton):
        self.cfg.set_config_setting('recon_select_on_image', recon_select_on_image_var.get())
        self.cfg.set_config_setting('recon_all_the_same', recon_all_the_same_var.get())
        if not recon_select_on_image_var.get() and recon_all_the_same_var.get():
            state = tk.NORMAL
            self.cfg.set_config_setting('recon_corners', ((0,0),(0,0)))
        else:
            state = tk.DISABLED
            self.cfg.set_config_setting('recon_corners', None)
        for corner in ['xmin', 'xmax', 'ymin', 'ymax']:
            getattr(self, f"recon_{corner}_var").set(0)
            corner_entries[corner]['state'] = state
        self.toggle_recon(recon_var, recon_all_the_same_checkbutton, recon_select_on_image_checkbutton, corner_entries)
        
    def update_recon_corners(self):
        xmin = self.recon_xmin_var.get()
        xmax = self.recon_xmax_var.get()
        ymin = self.recon_ymin_var.get()
        ymax = self.recon_ymax_var.get()
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

    ###################################### Reconstruction TAB ######################################
    # def create_delta_tab(self):
    #     self.create_delta_core_frame(self.delta_tab).pack(pady=10)
    
    # ######### Delta TAB: core frame #########
    # def create_delta_core_frame(self, tab):
    #     frame = ttk.Frame(tab)
    #     core_frame = ttk.Frame(frame)
        
    #     delta_bacteria_core_var = tk.StringVar(value=self.cfg.get_config_setting('delta_bacteria_core'))
    #     setattr(self, 'delta_bacteria_core_var', delta_bacteria_core_var)
    #     combobox = ttk.Combobox(core_frame, textvariable=delta_bacteria_core_var, values=['PH', 'BF'], width=5)
    #     combobox.bind("<<ComboboxSelected>>", lambda event: self.update_bacteria_core())
    #     tk.Label(core_frame, text='Core caluclated on').pack(side='left', padx=5)
    #     combobox.pack(side='left', padx=5)
    #     core_frame.pack()
        
    #     core_type = delta_bacteria_core_var.get().lower()
    #     var = tk.StringVar(value=self.cfg.get_config_setting(f'model_file_{core_type}_core_seg'))
    #     setattr(self, f'model_file_{core_type}_core_seg_var', var)
    #     entry = tk.Entry(frame, textvariable=var, width=80)
    #     self.bind_events(entry, ("<FocusOut>", "<Return>"), lambda event: self.update_file_path(f'model_file_{core_type}_core_seg'))
        
    
    # def update_file_path(self, config_variable):
    #     path = os.path.normpath(getattr(self, f"{config_variable}_var").get())
    #     if os.path.isfile(path):
    #         getattr(self, f"{config_variable}_var").set(path)
    #         self.cfg.set_config_setting(config_variable, path)
    #     else:
    #         tk.messagebox.showerror("Error", "File does not exist")
    #         self.base_dir_entry_var.set(self.cfg.get_config_setting(config_variable))
            
            
    # def select_file(self, config_variable):
    #     selected_folder = filedialog.askdirectory()
    #     if selected_folder:
    #         normalized_path = os.path.normpath(selected_folder)
    #         entry_var = getattr(self, f"{config_variable}_var")
    #         entry_var.set(normalized_path)
    #         self.cfg.set_config_setting(config_variable, normalized_path)
    #         if config_variable == 'base_dir':
    #             self.update_saving_dir_from_base_dir()
        
        
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
















