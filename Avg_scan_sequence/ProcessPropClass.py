# -*- coding: utf-8 -*-
"""
Created on 2023.01.23
Modified 2023.02.09 : autofocus implementation
Modified 2023.02.28 correction bug: list init not in the right place

@author: tcolomb
"""
import binkoala
import numpy as np
import os
from matplotlib import pyplot as plt
import skimage.restoration as skir
import numpy.ma as ma
import shutil
import tkinter
import preprocessing, cplxprocessing, FFTClass

from tkinter import *


class takeInput(object):
    def __init__(self,requestMessage,boolTextOrNumber,defaultText,hideText):
        self.root = Tk()
        self.string = ''
        self.frame = Frame(self.root)
        self.frame.pack()        
        self.acceptInput(requestMessage,defaultText,hideText)
        

    def acceptInput(self,requestMessage,defaultText,hideText):
        r = self.frame
        k = Label(r,text=requestMessage)
        k.pack(side='left')
        self.e = Entry(r,text='Name')
        if hideText:
            self.e["show"]="*"
        self.e.pack(side='left')
        self.e.insert(0,defaultText)
        self.e.focus_set()
        b = Button(r,text='okay',command=self.gettext)
        b.pack(side='right')

    def gettext(self):
        self.string = self.e.get()
        self.root.destroy()

    def getString(self):
        return self.string

    def waitForInput(self):
        self.root.mainloop()



###Class for complex averaging
class propagation_avg:
    def __init__(self, lambda_m, d_rec_m, CCD_px_size_m, unwrap=False,define_rec_dist=False, ROI=None, autofocus=False, autofocus_each_time_line=False, d_init_cm=0, range_focus_um=5, focusMethod=1):
        
        self.FFT = FFTClass.FFT(True)#class for the Fourier transform
        self.lambda_m = lambda_m #DHM wavelength, cannot be deduced from phase image because hconv depends also on RI
        self.CCD_px_size_m = CCD_px_size_m #CCD pixel size of the camera 
        self.d_rec_m = d_rec_m #propagation distance in meter
        self.define_rec_dist = define_rec_dist #define reconstruction distance for each well
        self.d_array_cm = None #array of the reconstruction distance for each array
        self.ROI = ROI #Region of interest size defined by the user
        self.autofocus = autofocus #Perform autofocus
        self.autofocus_each_time_line = autofocus_each_time_line #Perform autofocus for each timeline
        self.focusMethod = focusMethod #0=min std #1 Louis method
        self.d_init_cm = d_init_cm #initial reconstruction distance FROM the acquired data
        self.range_focus_um = range_focus_um #range for the autofocus
        self.stack_array = None #stack of the reconstructed wavefront for autofocus
        self.NA = 0.5 #average NA of objective, ideally should be the right one

        self.apodization = None #class for apodiation
        self.padd_data = False #boolean if the data have to be padded or not
        self.propagation = None #class for propagation
        self.apodiz_phase = False # apodization of the phase if initial data are not power of 2, defaut=False


        self.list_directory_ph = None #list of the phase data to propagate
        self.list_directory_amp = None #list of the amplitude to propagate
        self.amp_exist = True #by default is True, if amplitude does not exist -> False
        self.width = None #width of images
        self.height = None #height of images
        self.width_rec = None #has to be power of two
        self.height_rec = None #has to be power of two
        self.px_size = None #px_size in meter
        self.hconv_amp = None #hconv of amplitude image
        self.hconv_ph = None #hconv of phase image
        self.unit_code_amp = None #unit code for amplitude
        self.unit_code_ph = None #unit code for phase

        self.unwrap = unwrap #unwrapp the averaged phase if True
        self.directory_to_save = None #directory to save the data = directory_avg_d=
        self.directory = None
        
        
    def Open_Directory(self, directory, message):
        '''
        Open directory
        '''
        root = tkinter.Tk("Open Directory")
        fname = tkinter.filedialog.askdirectory(initialdir=directory, title="Open Directory")
        root.destroy()
        self.directory = fname
        
    def Set_Directory_input(self, directory):
        '''
        Set the directory automatically from the averaged directory for example
        '''
        self.directory = directory
        
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def set_bin_params(self, header_ph, header_amp=None):
        """
        From headers of amplitude and phase .bin file define the parameters
        Width and Height are used to define the reconsctruction width_rec and height_rec
        that should be power of two to apply propagation
        if the width and widht_rec are different the wavefront is apodized and padded
        """
        self.width = (int)(header_ph["width"]) #width of phase
        self.height = (int)(header_ph["height"]) #height of phase
        self.px_size = (float)(header_ph["px_size"]) #reconstructed pixel size in meter
        self.hconv_ph = (float)(header_ph["hconv"])#conversion factor of the phase (depends on lambda and RI)
        self.unit_code_ph = (int)(header_ph["unit_code"])
        if header_amp is not None:
            self.hconv_amp = (float)(header_amp["hconv"])
            self.unit_code_amp = (int)(header_amp["unit_code"])
        
        #Define the effective magnification from CCD px size and reconstruced px size
        self.MO = self.CCD_px_size_m/self.px_size
        self.dof_cm = self.lambda_m/(self.NA**2)*1e2/5 #approximative depth of focus /5 to be more precise
        self.d_range = np.abs(self.MO**2*self.range_focus_um)*1e-4 #convert range um to cm and to reconstruced distance
        self.d_step = self.MO**2*self.dof_cm     # reconstruction distance step from depth of focus  
        self.d_range *= 2
        
        #Define the next power of 2 size
        height_rec = self.nextpower(self.height,2)
        width_rec = self.nextpower(self.width,2)
        self.height_rec = np.max([height_rec, width_rec])
        self.width_rec = self.height_rec
        #set the propagation class with
        self.set_propagation_class()

    def set_propagation_class(self):
        """
        Init the apodization class (if necessary) and  the propagation class
        """
        if self.width != self.width_rec or self.height != self.height_rec:
            self.padd_data = True
            self.apodization = preprocessing.apodization(self.width, self.height) #init the apodization class
        self.propagation = cplxprocessing.fresnel_propagation(self.width_rec, self.height_rec, 1,self.d_rec_m,self.CCD_px_size_m,self.lambda_m,True)
    def getEntry(self, requestMessage,boolTextOrNumber,defaultText):
        """
        Used to get scalar value, if the user write a non scalar value, it returns infinity
        """
        msgBox=takeInput(requestMessage,boolTextOrNumber,defaultText, False)
        msgBox.waitForInput()
        if boolTextOrNumber: #True=text, False=Number
            return msgBox.getString()
        else:
            try:
                float(msgBox.getString())
                return float(msgBox.getString())
            except:
                return np.inf
    
    
    def define_d(self, w, well_number):
        """
        Define the reconstruction distance by asking the user to enter a reconstruction distance
        It goes out when the user click OK with the same reconstruction distance than previous value
        """
        response = "Continue"
        d_rec = self.d_rec_m*1e2
        propagate = True
        while response == "Continue":
            if propagate:
                self.propagation.set_rec_total(d_rec*1e-2)
                if self.padd_data and d_rec != 0:
                    w_prop = self.padding_wavefront(w)
                else:
                    w_prop = np.copy(w)
                w_prop = self.propagation(self.FFT.fft2(w_prop))
                w_prop = self.crop_data(w_prop)
                plt.figure(1)
                plt.imshow(np.angle(w_prop), cmap = "gray")
            d_user = self.getEntry("Enter a new reconstruction distance in cm if not in focus for "+str(well_number)+"th well", False, d_rec)
            if np.isscalar(d_user):
                if d_user != d_rec:
                    response = "Continue"
                    if not np.isinf(d_user):
                        d_rec = d_user
                        propagate = True
                        plt.close("all")
                    else:
                        propagate = False
                else:
                    response = "OK"
            else:
                # d_user is inf, error of the user
                #redo using the last value
                response = "Continue"
                propagate = False                           
        plt.close("all")
        self.d_rec_m = d_user*1e-2
        return d_user
    def set_unwrap(self, unwrap):
        self.unwrap = unwrap
        
    def save_ph_avg(self, fname, ph):
        height, width = np.shape(ph)
        binkoala.write_mat_bin(fname, ph, width, height, self.px_size, self.hconv_ph, self.unit_code_ph)
    
    def save_amp_avg(self, fname, amp):
        height, width = np.shape(amp)
        binkoala.write_mat_bin(fname, amp, width, height, self.px_size, self.hconv_amp, self.unit_code_amp)
    
    def crop_data(self, data):
        """
        Crop the wavefront to initial size if size not a power of 2, not necessary
        if the propagation distance is zero (no propagation)
        """
        if self.padd_data and self.propagation.rec_total != 0:
            top = max(0,(int)((self.height_rec-self.height)/2))
            left = max(0,(int)((self.width_rec-self.width)/2))
            data = data[top:top+self.height,left:left+self.width]
        return data
    
    def crop_data_from_ROI(self,data):
        '''
        Crop the data from the ROI defined by the user
        '''
        size = np.shape(data)
        if self.ROI is not None:
            top = max(0,(int)((size[0]-self.ROI)/2))
            left = max(0,(int)((size[1]-self.ROI)/2))
            data = data[top:top+self.ROI,left:left+self.ROI]
        return data
        
    def save_data(self, data, k):
        """
        Save the amplitude and phase images in .bin format
        data is the wavefront, k is the timeline value 00000, 00001,...
        """
        data = self.crop_data(data)
        data = self.crop_data_from_ROI(data)
        if k == 0:
#            utils.create_directory(self.directory_to_save+"Phase\\Float\\Bin\\")
            self.create_directory(self.directory_to_save+"Phase")
            self.create_directory(self.directory_to_save+"Phase\\Float")
            self.create_directory(self.directory_to_save+"Phase\\Float\\Bin")
        ph = np.angle(data)
        if self.unwrap:
            ph = self.get_result_unwrap(ph)
        fname = self.directory_to_save+"Phase\\Float\\Bin\\"+str(k).zfill(5)+"_phase.bin"
        self.save_ph_avg(fname, ph)
        if self.amp_exist and k == 0:
#            utils.create_directory(self.directory_to_save+"Intensity\\Float\\Bin\\")
            self.create_directory(self.directory_to_save+"Intensity")
            self.create_directory(self.directory_to_save+"Intensity\\Float")
            self.create_directory(self.directory_to_save+"Intensity\\Float\\Bin")
        if self.amp_exist:   
            amp = np.abs(data)
            fname = self.directory_to_save+"Intensity\\Float\\Bin\\"+str(k).zfill(5)+"_intensity.bin"
            self.save_amp_avg(fname, amp)
        
    
    def get_result_unwrap(self, phase, mask=None):
        """
        Perform path following unwrap
        """
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))
    
    def nextpower(self, num, base):
        """
        Define the next power of 2 size
        """
        i = 1
        while i < num: i *= base
        return i
    def padding_wavefront(self, w):
        """
        Padding the wavefront to propagate with wavefront size power of 2
        """
        size = np.shape(w)
        if size != (self.height_rec, self.width_rec):
            amp = np.abs(w)
            ph = np.angle(w)
            amp = self.apodization(amp)
            if self.apodiz_phase:
                ph = self.apodization(ph)
            w = amp*np.exp(complex(0.,1.)*ph)
            w_out = np.zeros((self.height_rec, self.width_rec))*complex(1.,0)
            x = (int)((self.height_rec-np.shape(w)[0])/2)
            y = (int)((self.width_rec-np.shape(w)[1])/2)
            w_out[x:x+size[0],y:y+size[1]]= w
            return w_out
        else:
            return w
    
    
    def set_directory(self):
        """
        Set directory
        Define the list of amplitude and phas to reconctruct for each well
        Define the reconstruction distance for each well if asking (self.define_rec_dist)
        """
        sequence_name = os.path.basename(os.path.normpath(self.directory))
        parent_dir = os.path.dirname(self.directory)
        ###test if wrong directory (should not have experimental_info.json)
        continue_code = os.path.exists(os.path.normpath(self.directory)+"\\experimental_info.json")
        continue_code = not continue_code
        if not continue_code:
            print("The directory is not an averaged directory as experimental_info.json exists in it")
        else:
            list_wells = []
            list_directory_ph = []
            list_directory_amp = []
            list_well = [ name for name in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, name)) ]
            list_well_real = []
            for well in list_well:
                test = well[0:5]
                if test.isnumeric():
                    list_well_real.append(well)
            list_well = list_well_real
            if len(list_well) == 0:
                list_wells = [self.directory]
            else:
                list_wells = []
                for well in list_well:
                    list_wells.append(self.directory+"\\"+well)
            self.directory_timestamps = list_wells
            
    #        continue_code = True
            list_timeline = [ name for name in os.listdir(list_wells[-1]+"\\Phase\\Float\\Bin") ]
            try:
                list_timeline = [ name for name in os.listdir(list_wells[-1]+"\\Phase\\Float\\Bin") ]
            except:
                print("There is no data to analyze")
                continue_code = False
            if continue_code:          
                self.amp_exist = True
                time_amp = list_timeline[0][:-9]+"intensity.bin"
                if not os.path.exists(list_wells[0]+"\\Intensity\\Float\\Bin\\"+time_amp):
                    self.amp_exist = False
      
                self.list_well_to_perform_ph = []
                self.list_well_to_perform_amp = []
                self.directory_to_save_well = []
                only_a_single_well = np.shape(list_wells)[0]==1
                
                self.d_array_cm = [] #array of the reconstruction distance in cm if asking to define
                for k, well in enumerate(list_wells):
                    if only_a_single_well:
                        self.directory_to_save_well.append(parent_dir+"\\"+sequence_name+"_prop\\")
                    else:
                        self.directory_to_save_well.append(parent_dir+"\\"+sequence_name+"_prop\\"+os.path.basename(well))
                    list_directory_ph = []
                    list_directory_amp = []
                    for time_ph in list_timeline:
                        if self.amp_exist:
                            time_amp = time_ph[:-9]+"intensity.bin"
        
                        list_directory_ph.append(well+"\\Phase\\Float\\Bin\\"+time_ph)
                        if self.amp_exist:
                            list_directory_amp.append(well+"\\Intensity\\Float\\Bin\\"+time_amp)
                        
                    self.list_well_to_perform_ph.append(np.copy(list_directory_ph))
                    if self.amp_exist:
                        self.list_well_to_perform_amp.append(np.copy(list_directory_amp))
                     
                    ## define binary parameters
                    ph, header_ph = binkoala.read_mat_bin(self.list_well_to_perform_ph[0][0])
    
                    if self.amp_exist:
                        amp, header_amp = binkoala.read_mat_bin(self.list_well_to_perform_amp[0][0])
                    else:
                        header_amp = None
                    self.set_bin_params(header_ph, header_amp)
                    ##define rec_dist

                    if self.define_rec_dist:
                        if not self.autofocus:
                            if k == 0:
                                print("Define reconstruction distance")
                            w = self.cstr_cplx_wavefront(self.list_well_to_perform_ph[k][0],self.list_well_to_perform_amp[k][0])
                            d = self.define_d(w,k+1)
                            self.d_array_cm.append(d)
                        else:
                            if k ==0:
                                print("Perform autofocus for first image of all wells")
                            d_focus = self.autofocus_process(self.list_well_to_perform_ph[k][0], self.list_well_to_perform_amp[k][0])
                            print(d_focus)
                            self.d_array_cm.append(d_focus)
        return continue_code
                

    def cstr_cplx_wavefront(self,ph_name, amp_name=None):
        """
        Construction of wavefront from phase and amplitude name
        """
        ph, header = binkoala.read_mat_bin(ph_name)
        w = np.exp(complex(0.,1.)*ph)
        if self.amp_exist:
            amp, header = binkoala.read_mat_bin(amp_name)
            w *= amp
        return w
    
        
    def propagation_process(self, ph_name, amp_name, d=None):
        """
        Propagate the wavefront only if reconstruction distance not zero
        """
        
        if d is not None:
            self.propagation.set_rec_total(d*1e-2)
        w = self.cstr_cplx_wavefront(ph_name, amp_name)
        if self.propagation.rec_total != 0:
            if self.padd_data:
                w = self.padding_wavefront(w)
            w = self.FFT.fft2(w) #propagation is defined from FFT of the wavefront
            w = self.propagation(w)
        return w
    
    def autofocus_process(self, ph_name, ph_amp):
        """
        Process for autofocus
        """        
        d_min = self.d_init_cm-self.d_range/2
        d_max = self.d_init_cm+self.d_range/2
        number = (int)(self.d_range/self.d_step)+1
        d_array = np.linspace(d_min,d_max, number)
        for k, d in enumerate(d_array):
            w_propagate = self.propagation_process(ph_name, ph_amp,d)
            w_propagate = self.crop_data(w_propagate)
            w_propagate = self.crop_data_from_ROI(w_propagate)
            if k==0:
                self.stack_array = np.zeros((np.shape(w_propagate)[0],np.shape(w_propagate)[1],number))*complex(1.0,0.)
            self.stack_array[:,:,k]= w_propagate
        d_focus = self.Find_d_focus(d_array)
        return d_focus
        
    def Find_d_focus(self, d_array):
        """
        Determine the focus position (reconstruction distance) from the wavefront stack
        """
        if self.focusMethod == 0: #minimal std
            img_array = np.abs(self.stack_array)
            m = np.mean(img_array, axis=(0,1))
            fx = np.std(img_array, axis=(0,1))/m
            index = fx.argmin()
            d = d_array[index]
            
        if self.focusMethod == 1: #Louis method
            amp_array = np.abs(self.stack_array)
            ph_array = np.angle(self.stack_array)
            fx = -np.std(amp_array, axis=(0,1))
            fx = fx - fx.min()
            fx2 = np.std(ph_array, axis=(0,1))
            fx *= fx2-fx2.min()
            index = fx.argmax()
            d = d_array[index]
#        plt.figure(1)
#        plt.plot(d_array, fx)
        return d
        
    
    def process_all_wells(self):
        """
        Process the data well by well
        """
        Number_well = len(self.directory_to_save_well)
        for k in range(Number_well):
            if self.define_rec_dist:
                if not (self.autofocus and self.autofocus_each_time_line):
                    d_m = self.d_array_cm[k]*1e-2
                    if d_m != self.d_rec_m:
                        self.propagation.set_rec_total(d_m)
                        self.d_rec_m = d_m
            self.directory_to_save = self.directory_to_save_well[k]
            if Number_well != 0:
                self.directory_to_save += "\\"
            self.list_directory_ph = self.list_well_to_perform_ph[k]
            if self.amp_exist:
                self.list_directory_amp = self.list_well_to_perform_amp[k]
            self.process(k, Number_well)
            fname_timestamps_original = self.directory_timestamps[k]+"\\timestamps.txt"
            fname_timestampsForCAT_original = self.directory_timestamps[k]+"\\timestampsForCAT.txt"            
            if os.path.exists(fname_timestamps_original):
                target = self.directory_to_save_well[k]+"\\timestamps.txt"
                shutil.copyfile(fname_timestamps_original, target)
            if os.path.exists(fname_timestampsForCAT_original):
                target = self.directory_to_save_well[k]+"\\timestampsForCAT.txt"
                shutil.copyfile(fname_timestampsForCAT_original, target)
            
              
    
    def process(self, N_well = 0, N_well_total=1):
        """
        Process all the timeline data for a given well, propagate and save
        """
        it = 0
        Number = np.shape(self.list_directory_ph)[0]
        for k, list_ph in enumerate(self.list_directory_ph):
            if self.autofocus and self.autofocus_each_time_line:
                self.d_init_cm = self.d_array_cm[N_well]
                d_focus_cm = self.autofocus_process(self.list_directory_ph[k], self.list_directory_amp[k])
                print((int)(d_focus_cm*1000)/1000)
                self.propagation.set_rec_total(d_focus_cm*1e-2)
                self.d_rec_m = d_focus_cm*1e-2
                
            w = self.propagation_process(self.list_directory_ph[k], self.list_directory_amp[k])
            self.save_data(w,k)
            it += 1
            print(str((int)(it/Number*100))+"% of "+str(N_well+1)+"/"+str(N_well_total)+" wells")
        

            
        
