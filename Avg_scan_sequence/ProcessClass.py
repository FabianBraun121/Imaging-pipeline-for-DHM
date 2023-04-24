# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:19:38 2020
Modified 2023.03.22 : Add computation shift with unwrapped phase

@author: tcolomb
"""
import binkoala
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
try:
    from skimage.registration import  phase_cross_correlation
    use_phase_cross_correlation = True
except:
    from skimage.feature import register_translation
    use_phase_cross_correlation = False
import skimage.restoration as skir
import numpy.ma as ma
import shutil
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import csv
import tkinter
from tkinter import filedialog, simpledialog, messagebox


#def Open_Directory(directory, message):
#    root = tkinter.Tk("Open Directory")
#    fname = tkinter.filedialog.askdirectory(initialdir=directory, title="Open Directory")
#    root.destroy()
#    return fname

##Class for GaussFit to measure offset
class GaussFit:
    def __init__(self, sigma):
        self.sigma = sigma
    def GaussianFit_fixed_sigma(self, x,coherence):
        index_max = np.argmax(coherence)
        A_not_fitted = np.max(coherence)
        mu_not_fitted = x[index_max]
        p0=[A_not_fitted, mu_not_fitted]
        try:
            coeff,var_matrix = curve_fit(self.gaussfixedsigma_px, x, coherence, p0=p0)
            return coeff
        except:
            return p0
    def gaussfixedsigma_px(self, x,*p):
            A, mu = p
            return A*np.exp(-(x-mu)**2/(2.*self.sigma**2))
    def GaussianFit_with_sigma(self, x,coherence):
        index_max = np.argmax(coherence)
        A_not_fitted = np.max(coherence)
        mu_not_fitted = x[index_max]
        sigma2 = self.sigma
        p0=[A_not_fitted, mu_not_fitted, sigma2]
#        print(p0)
        try:
            coeff,var_matrix = curve_fit(self.gausswithsigma_px, x, coherence, p0=p0)
            return coeff
        except:
            print("error")
            return p0
    def gausswithsigma_px(self, x,*p):
            A, mu, sigma2 = p
            return A*np.exp(-(x-mu)**2/(2.*sigma2**2))

###Class for complex averaging
class complex_averaging:
    def __init__(self,shift_timeline=False, Use_ROI=False, use_amplitude_for_shift=False, UseScanValue=False, unwrap=True, use_background=False, use_unwrap_for_shifting =False):
        self.GaussFitClass = GaussFit(0.05)#Gauss fit to measure offset
        self.list_directory_ph = None #list of the phase data to average 
        self.list_directory_amp = None #list of the amplitude data to average
        self.use_amp_for_shift = use_amplitude_for_shift #use only amplitude to measure the shift between images
        self.amp_exist = True #by default is True, if amplitude does not exist -> False
        self.width = None #width of images
        self.height = None #height of images
        self.px_size = None #px_size in meter
        self.hconv_amp = None #hconv of amplitude image
        self.hconv_ph = None #hconv of phase image
        self.unit_code_amp = None #unit code for amplitude
        self.unit_code_ph = None #unit code for phase
        self.refimage = None #reference image to measure the shift, can be amplitude or phase and define with a ROI
        self.background = None ## not used, always None
        self.compute_background = False
        self.gaussian_value_background = 5 #seems good value
        self.use_background = use_background
        self.ROI = None #ROI for a given well (define by ROI_array)
        self.ROI_array = None #ROI array, a different ROI for the different wells
        self.use_ROI = Use_ROI #use ROI, if False use the entire image
        self.UseScanValue = UseScanValue #use the stage position from timestamps instead of computing the shift
        self.x_ref = None #x reference position if UseScanValue
        self.y_ref = None #y reference position if UseScanValue
        self.x = None #actual x position of the stage
        self.y = None #actual y position of the stage
        self.cplx_avg = None #complex averaging
        self.unwrap = unwrap #unwrapp the averaged phase if True
        self.directory_to_save = None #directory to save the data = directory_avg
        self.mask = None #mask used for the averaging
        self.overlapped_images = None #used to perform the averaging /overlapped_images
        self.shift_time_line = shift_timeline #perform shift along the timeline
        self.mode_avg = None
        self.directory = None
        self.set_mode_avg()
        self.unwrap_phase_for_shift = True
        
    def Open_Directory(self, directory, message):
        root = tkinter.Tk("Open Directory")
        fname = tkinter.filedialog.askdirectory(initialdir=directory, title="Open Directory")
        root.destroy()
        self.directory = fname
    def set_mode_avg(self):
        if self.UseScanValue:
            self.mode_avg = "scan"
            self.use_background = False
        else:
            if self.use_amp_for_shift:
                self.mode_avg = "amp"
            else:
                self.mode_avg = "ph"
            if self.use_background:
                self.mode_avg += "_bck"
        
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def get_directory_to_save(self):
        return os.path.normpath(self.directory_to_save)
    
    def set_bin_params(self, header_ph, header_amp=None):
        self.width = (int)(header_ph["width"])
        self.height = (int)(header_ph["height"])
        self.px_size = (float)(header_ph["px_size"])
        self.hconv_ph = (float)(header_ph["hconv"])
        self.unit_code_ph = (int)(header_ph["unit_code"])
        if header_amp is not None:
            self.hconv_amp = (float)(header_amp["hconv"])
            self.unit_code_amp = (int)(header_amp["unit_code"])
    def set_use_scan_value(self, UseScanValue):
        self.UseScanValue = UseScanValue
        self.set_mode_avg()

    def set_use_background(self, use_background):
        self.use_background = use_background
        self.set_mode_avg()
        self.set_directory()
    def set_use_ROI(self, use_ROI):
        self.use_ROI = use_ROI
        
    def set_shift_timeline(self, shift_time_line):
        self.shift_time_line = shift_time_line
    
    def set_unwrap(self, unwrap):
        self.unwrap = unwrap
        
    def save_ph_avg(self, fname, ph):
        binkoala.write_mat_bin(fname, ph, self.width, self.height, self.px_size, self.hconv_ph, self.unit_code_ph)
    
    def save_amp_avg(self, fname, amp):
        binkoala.write_mat_bin(fname, amp, self.width, self.height, self.px_size, self.hconv_amp, self.unit_code_amp)
    
    def save_data(self, data, k): #save with usual sequence format !
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
        ph_m = ma.array(phase, mask=mask)
        return np.array(skir.unwrap_phase(ph_m))
    
    def set_use_amplitude_for_shift(self, use_amp_for_shift):
        self.use_amp_for_shift = use_amp_for_shift
        self.set_mode_avg()
    
    def read_position_timestamps(self, list_ph):
        directory_name = list_ph
        time = os.path.basename(directory_name)
        time = (int)(time[:-10])

        for i in range(4):
            directory_name = os.path.dirname(directory_name)
        timestamps_file = directory_name+"\\timestamps.txt"
        data = self.read_timestamps(timestamps_file)
        x = (float)(data[time,4]) #x stage position
        y = (float)(data[time,5]) #y stage position
#        z = (float)(data[time,6]) #z stage position (NOT USED)
        return x,y
        
        
    def read_timestamps(self, fname):
        data = []
        with open(fname, 'r') as f:
            csv_reader = csv.reader(f, delimiter=' ')
            for row in csv_reader:
                if np.shape(row)[0] == 6:
                    data.append([(int)(row[0]),row[1],row[2],(float)(row[3]),(float)(row[4]), (float)(row[5]),0])
                else:
                    data.append([(int)(row[0]),row[1],row[2],(float)(row[3]),(float)(row[4]), (float)(row[5]),(float)(row[6])])
        data = np.array(data)
        return data
    
    def set_directory(self):
        sequence_name = os.path.basename(os.path.normpath(self.directory))
        parent_dir = os.path.dirname(self.directory)
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
        
        list_pos_by_well = [ name for name in os.listdir(self.directory+"\\"+list_well[0]) if os.path.isdir(os.path.join(self.directory+"\\"+list_well[0], name)) ]
        there_is_position = False
        for pos in list_pos_by_well:
            test = pos[0:5]
            if test.isnumeric():
                there_is_position = True
        if there_is_position:
            list_wells = []
            for well in list_well:
                list_wells.append(self.directory+"\\"+well)
        else:
            list_wells = [self.directory]
            list_pos_by_well = list_well
        
        
        N_well_x, N_well_y = (int)(list_pos_by_well[-1][0:5]),(int)(list_pos_by_well[-1][6:])
        #take the middle image as first one
        pos = (int)(N_well_y/2)*N_well_x+(int)(N_well_y/2)
        directory_timestamps = list_pos_by_well[pos]
        
        list_pos_by_well.insert(0, list_pos_by_well[pos])
        del list_pos_by_well[(int)(pos+1)]
        list_pos_by_well = list_pos_by_well
        #define time line number list:
        continue_code = True
        try:
            list_timeline = [ name for name in os.listdir(list_wells[-1]+"\\"+list_pos_by_well[-1]+"\\Phase\\Float\\Bin") ]
        except:
            print("There is no data to analyze")
            continue_code = False
        if np.shape(list_timeline)[0] == 0:
            continue_code = False
            print("No data in", list_wells[-1]+"\\"+list_pos_by_well[-1]+"\\Phase\\Float\\Bin")
        if continue_code:   
#            list_timeline = list_timeline           
            self.amp_exist = True
            time_amp = list_timeline[0][:-9]+"intensity.bin"
            if not os.path.exists(list_wells[0]+"\\"+list_pos_by_well[0]+"\\Intensity\\Float\\Bin\\"+time_amp):
                self.amp_exist = False
                self.use_amp_for_shift = False #if intensisty does not exit cannot use amplitude as criteria
                self.set_mode_avg()
                if not self.UseScanValue:
                    print("Not possible to use amplitude as criteria for the shift determination as there is no intensity data")
  
            self.list_well_to_perform_ph = []
            self.list_well_to_perform_amp = []
            self.directory_to_save_well = []
            self.ROI_array = []
            self.directory_timestamps = []
            only_a_single_well = np.shape(list_wells)[0]==1
            for well in list_wells:
                self.directory_timestamps.append(well+"\\"+directory_timestamps)
                if only_a_single_well:
                    self.directory_to_save_well.append(parent_dir+"\\"+sequence_name+"_avg_"+self.mode_avg+"\\")
                else:
                    self.directory_to_save_well.append(parent_dir+"\\"+sequence_name+"_avg_"+self.mode_avg+"\\"+os.path.basename(well))
                list_to_average_ph = []
                list_to_average_amp = []
                for time_ph in list_timeline:
                    if self.amp_exist:
                        time_amp = time_ph[:-9]+"intensity.bin"
    
                    list_directory_ph = []
                    list_directory_amp = []
                    for pos in list_pos_by_well:
                        list_directory_ph.append(well+"\\"+pos+"\\Phase\\Float\\Bin\\"+time_ph)
                        if self.amp_exist:
                            list_directory_amp.append(well+"\\"+pos+"\\Intensity\\Float\\Bin\\"+time_amp)
                    list_to_average_ph.append(np.copy(list_directory_ph))
                    if self.amp_exist:
                        list_to_average_amp.append(np.copy(list_directory_amp))
                    
                self.list_well_to_perform_ph.append(np.copy(list_to_average_ph))
                if self.amp_exist:
                    self.list_well_to_perform_amp.append(np.copy(list_to_average_amp))
                        
                ## define binary parameters
                ph, header_ph = binkoala.read_mat_bin(list_to_average_ph[0][0])
                ##define ROI
                if self.use_ROI:
                    ROI = self.define_ROI(ph)
                    self.ROI_array.append(ROI)
                if self.amp_exist:
                    amp, header_amp = binkoala.read_mat_bin(list_to_average_amp[0][0])
                else:
                    header_amp = None
                self.set_bin_params(header_ph, header_amp)
        return continue_code
                

    def test_data(self):
        Number_well = len(self.directory_to_save_well)
        ph_corrupted = []
        amp_corrupted = []
        for k in range(Number_well):
            self.list_directory_ph = self.list_well_to_perform_ph[k]
            for fname in self.list_directory_ph:
                for ph_name in fname:
                    size = os.path.getsize(ph_name)
                    if size == 0:
                        ph_corrupted.append(ph_name)
                    else:
                        try:
                            ph, header = binkoala.read_mat_bin(ph_name)
                        except:
                            ph_corrupted.append(ph_name)
                    
            if self.amp_exist:
                self.list_directory_amp = self.list_well_to_perform_amp[k]
                for fname in self.list_directory_amp:
                    for amp_name in fname:
                        size = os.path.getsize(amp_name)
                        if size == 0:
                            amp_corrupted.append(amp_name)
                        else:
                            try:
                                ph, header = binkoala.read_mat_bin(amp_name)
                            except:
                                amp_corrupted.append(amp_name)
            else:
                amp_corrupted = None
        return ph_corrupted, amp_corrupted
            
        
    
    def cstr_cplx_wavefront(self,k, list_ph, list_amp=None):
        ph_name = list_ph[k]
        ph, header = binkoala.read_mat_bin(ph_name)
        w = np.exp(complex(0.,1.)*ph)
        if self.amp_exist:
            amp_name = list_amp[k]
            try:
                amp, header = binkoala.read_mat_bin(amp_name)
            except:
                print(amp_name)
            w *= amp
        return w
    
    def define_ROI(self, ph):
        plt.figure()
        plt.imshow(ph)
        pts = plt.ginput(2)
        ROI_0 = (int)(pts[0][0])
        ROI_1 = (int)(pts[0][1])
        ROI_2 = (int)(pts[1][0])
        ROI_3 = (int)(pts[1][1])
        
        if ROI_1>ROI_3:
            tmp= np.copy(ROI_1)
            ROI_1 = ROI_3
            ROI_3 = tmp
        if ROI_0>ROI_2:
            tmp= np.copy(ROI_0)
            ROI_0 = ROI_2
            ROI_2 = tmp
        plt.close("all")
        self.ROI = [ROI_1,ROI_3,ROI_0,ROI_2]
        return self.ROI
    
    def compute_background_process(self, k, Number_well):
        self.compute_background = True
        it = 0
        background = np.zeros((self.height,self.width))*complex(0.,1.)
        Number = np.shape(self.list_directory_ph)[0]
        for k, list_ph in enumerate(self.list_directory_ph):
            it += 1
            avg = self.complex_avg_directory(self.list_directory_ph[k], self.list_directory_amp[k])                
            background += avg
            print("Background:"+str((int)(it/Number*100))+"%")
            
        background /= (k+1)
        
        self.background = gaussian_filter(np.real(background), sigma=self.gaussian_value_background)+complex(0.,1.)*gaussian_filter(np.imag(background),sigma=self.gaussian_value_background)
        self.compute_background = False
        
    def complex_avg_directory(self, list_ph, list_amp):
        w = self.cstr_cplx_wavefront(0, list_ph, list_amp)
        self.overlapped_images = np.ones((self.height,self.width))
        if not self.compute_background:
            if self.background is not None:
                w /= self.background
                
            if not self.use_amp_for_shift and self.amp_exist: #if amplitude exists but not used, use only the phase
                refimage = np.copy(np.angle(w))
            else:
                refimage = np.abs(w)
            if self.use_ROI:
                refimage = refimage[self.ROI[0]:self.ROI[1],self.ROI[2]:self.ROI[3]]
            self.refimage = refimage
            if self.UseScanValue:
                self.x_ref, self.y_ref = self.read_position_timestamps(list_ph[0])
        self.avg_cplx = np.copy(w)
        for k in range(np.shape(list_ph)[0]-1):
            w = self.cstr_cplx_wavefront(k+1, list_ph, list_amp)
            if not self.compute_background:
                if self.UseScanValue:
                    self.x, self.y = self.read_position_timestamps(list_ph[k+1])
                w, shiftVector = self.shift_wavefront(w)
            w = self.adjust_offset(w)
            self.overlapped_images +=  np.where(np.isnan(self.mask),0,1)
            self.avg_cplx += w
            if not self.compute_background :
                if self.background is not None:
                    refimage = self.avg_cplx/self.background/self.overlapped_images
                else:
                    refimage = np.copy(self.avg_cplx/self.overlapped_images)
                if not self.use_amp_for_shift and self.amp_exist:
                    refimage = np.copy(np.angle(refimage))
                else:
                    refimage = np.copy(np.abs(refimage))
                if self.use_ROI:
                    refimage = refimage[self.ROI[0]:self.ROI[1],self.ROI[2]:self.ROI[3]]
                self.refimage = refimage
        self.avg_cplx /= self.overlapped_images

        return self.avg_cplx
    
    def process_all_wells(self):
        Number_well = len(self.directory_to_save_well)
        for k in range(Number_well):
            if self.use_ROI:
                self.ROI = self.ROI_array[k]
            self.directory_to_save = self.directory_to_save_well[k]
            if Number_well != 0:
                self.directory_to_save += "\\"
            self.list_directory_ph = self.list_well_to_perform_ph[k]
            if self.amp_exist:
                self.list_directory_amp = self.list_well_to_perform_amp[k]
            if self.use_background:
                self.compute_background_process(k, Number_well)
            self.process(k, Number_well)
            fname_timestamps_original = self.directory_timestamps[k]+"\\timestamps.txt"
            fname_timestampsForCAT_original = self.directory_timestamps[k]+"\\timestampsForCAT.txt"            
            if os.path.exists(fname_timestamps_original):
                target = self.directory_to_save_well[k]+"\\timestamps.txt"
                shutil.copyfile(fname_timestamps_original, target)
            if os.path.exists(fname_timestampsForCAT_original):
                target = self.directory_to_save_well[k]+"\\timestampsForCAT.txt"
                shutil.copyfile(fname_timestampsForCAT_original, target)
            
              
    
    def process(self, N_well = 1, N_well_total=1):
        it = 0
#        self.apply_background = True
        Number = np.shape(self.list_directory_ph)[0]
        avg_data = []
        for k, list_ph in enumerate(self.list_directory_ph):
            avg = self.complex_avg_directory(self.list_directory_ph[k], self.list_directory_amp[k])
#            if self.apply_background:
#                avg /= self.background
            avg_data.append(avg)
            it += 1
            print(str((int)(it/Number*100))+"% of "+str(N_well+1)+"/"+str(N_well_total)+" wells")
            if not self.shift_time_line:
                self.save_data(avg,k)
        if self.shift_time_line:
            self.UseScanValue = False
            avg_data = self.shift_avg_data(avg_data)
            for k,avg in enumerate(avg_data):
                self.save_data(avg,k)
        
            
    def shift_avg_data(self, avg_data):
        avg_data_shift = []
        if self.background is not None:
            refimage = avg_data[0]/self.background
        else:
            refimage = np.copy(avg_data[0])
        if not self.use_amp_for_shift and self.amp_exist:
            refimage = np.copy(np.angle(refimage))
        else:
            refimage = np.copy(np.abs(refimage))
        if self.use_ROI:
            refimage = refimage[self.ROI[0]:self.ROI[1],self.ROI[2]:self.ROI[3]]
        self.refimage = refimage
        avg_data_shift.append(avg_data[0])
        for cplx in avg_data[1:]:
            cplx, shiftVector = self.shift_wavefront(cplx)
            ## as a cplx average exists as there is a reference image, we apply the offset on the previous avg to have same offset
            cplx = self.adjust_offset(cplx)
            avg_data_shift.append(cplx)
        return avg_data_shift
            
            
    def shift_wavefront(self, w):
        if self.UseScanValue:
            shiftVector = ((self.y-self.y_ref)/(self.px_size*1e6),(self.x-self.x_ref)/(self.px_size*1e6))

        else:
            w_to_shift = np.copy(w)
            if self.background is not None:
                w_to_shift /= self.background
            if not self.use_amp_for_shift and self.amp_exist: #if amplitude exists but not used, use only the phase
                w_to_shift = np.copy(np.angle(w_to_shift))
            else:
                w_to_shift = np.copy(np.abs(w_to_shift))
            if self.ROI is not None:
                w_to_shift = w_to_shift[self.ROI[0]:self.ROI[1],self.ROI[2]:self.ROI[3]]
            if self.unwrap_phase_for_shift and not self.use_amp_for_shift:
                ref = self.get_result_unwrap(self.refimage)
                w_shift = self.get_result_unwrap(w_to_shift)
            else:
                ref = self.refimage
                w_shift = w_to_shift
            if use_phase_cross_correlation:
                shift_measured, error, diffphase = phase_cross_correlation(ref, w_shift, upsample_factor=10)
            else:
                shift_measured, error, diffphase = register_translation(ref, w_shift, upsample_factor=10)
            shiftVector = (shift_measured[0],shift_measured[1])

        #interpolation to apply the computed shift (has to be performed on float array)
        mask = ndimage.interpolation.shift(np.real(w), shift=shiftVector, mode='constant', cval = np.nan)
        self.mask = np.where(np.isnan(mask),mask,1)
        real = ndimage.interpolation.shift(np.real(w), shift=shiftVector, mode='constant', cval = 0)
        imaginary = ndimage.interpolation.shift(np.imag(w), shift=shiftVector, mode='constant', cval = 0)
        w = real+complex(0.,1.)*imaginary
        return w, shiftVector

    def adjust_offset(self, w):
        #perform complex averaging
        
        if self.mask is None:
            self.mask = np.ones(np.shape(w))
        if not self.compute_background:
            z= np.angle(np.multiply(w,np.conj(self.avg_cplx)))*self.mask #phase differenc between actual phase and avg_cplx phase
            z = z.ravel()
            z = z[np.logical_not(np.isnan(z))]
            #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
            hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
            x = np.linspace(np.min(z),np.max(z),1000)
            coefs = self.GaussFitClass.GaussianFit_with_sigma(x,hist[0])
            offset_value = coefs[1]
        
            s = np.std(z)
            signmap = (z < 0).astype(float)*2.*np.pi
            z2 = z + signmap
            s2 = np.std(z2)
            if (s2 < s - 0.05):
                hist = np.histogram(z2,bins=1000, range=(np.min(z2),np.max(z2)))
                x = np.linspace(np.min(z2),np.max(z2),1000)
                coefs = self.GaussFitClass.GaussianFit_with_sigma(x,hist[0])
                offset_value = coefs[1]
            w *= np.exp(-offset_value*complex(0.,1))#compensate the offset for the new wavefront
        return w
        
