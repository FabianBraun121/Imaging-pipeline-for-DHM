# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:46:55 2017

@author: tcolomb
"""

import numpy as np
import scipy

try:
    from reikna import cluda
    from reikna.fft import FFT as rFFT
    from reikna.fft import FFTShift as rFFTShift

except:
    print("pip install reikna and verify that it works with pyopencl 1.1!!!")
"""
Class for Fourier filtering in the processing
""" 
class FFT:
    def __init__(self, normalization=False):
#        self.gpuAvailable = False
        try:
            #InitReikna on Opencl
            api = cluda.ocl_api()
            platforms = api.get_platforms()  
            #find AMD or NVIDIA platforms
            for platform in platforms:
                if 'AMD'  in str(platform) or 'NVIDIA' in str(platform):
                    break;
            self.thr = api.Thread(platform.get_devices()[0])
            self.gpuAvailable = True
        except:
            self.gpuAvailable = False
        if self.gpuAvailable:
            self.useGPU = True
        else:
            self.useGPU = False
        #self.useGPU = True
        self.normalization = normalization
        #self.gpuAvailable = False
    def set_gpuAvailable(self, gpuAvailable):
        self.gpuAvailable = gpuAvailable
        #self.gpuAvailable = False
    def set_useGPU(self, useGPU):
        self.useGPU = useGPU
        if not self.gpuAvailable and self.useGPU:
            self.useGPU = False
            print("GPU is not available")
            
    def set_normalization(self, value):
        self.normalization = value
    def fft2(self, data):
#        print("fft2")
#        print("GPU=",self.useGPU)
#        print("Normalization=",self.normalization)
#        if self.useGPU:
##            print("GPU")
##            print("fft2")
##            test1 = np.fft.fft2(data)
##            if self.normalization:
##                size = np.shape(data)
##                test1 *= 1/(np.sqrt(size[0]*size[1])) 
##            print(self.normalization)
#            data = data.astype(np.complex64)
#            d_data = cuda.to_device(data)
#            fftplan = Plan.two(CUFFT_C2C, *data.shape)
#            fftplan.forward(d_data, d_data)
#            d_data.copy_to_host(data)
#            if self.normalization:
#                size = (np.shape(data))
#                data *= 1/(np.sqrt(size[0]*size[1])) 
##            test2=np.average(np.abs(test1/data))
##            print(test2)
#            return data
        size = np.shape(data)
        m = 1/(np.sqrt(size[0]*size[1]))
        if self.useGPU:
            data_gpu =  self.thr.to_device(data.astype(np.complex64))
            program = rFFT(data_gpu)
            fftc = program.compile(self.thr, fast_math=True)
            fftc(data_gpu,data_gpu,inverse=0)
            if self.normalization:
                data_gpu *= m
            signal_fft = data_gpu.get()
        else:
            signal_fft = np.fft.fft2(data)
#        size = np.shape(data)
            if self.normalization:
                signal_fft *= m
        return signal_fft

    def ifft2(self, data):
#        print("ifft")
#        print("GPU=",self.useGPU)
#        print("Normalization=",self.normalization)
#        if self.useGPU:
##            print("GPU")
##            print("ifft2")
##            test1 = np.fft.ifft2(data)
##            if self.normalization:
##                size = np.shape(data)
##                test1 *= (np.sqrt(size[0]*size[1]))
##            else:
##                size = np.shape(data)
##                test1 *= (size[0]-1)*(size[1]-1)
##            print(self.normalization)
#            data = data.astype(np.complex64)
#            d_data = cuda.to_device(data)
#            fftplan = Plan.two(CUFFT_C2C, *data.shape)
#            fftplan.inverse(d_data, d_data)
#            d_data.copy_to_host(data)
#            if self.normalization:
#                size = (np.shape(data))
##                print(size)
#                #data *= size[0]*size[1]/(np.sqrt(size[0]*size[1]))
#                data *= 1/(np.sqrt(size[0]*size[1]))
##                print("test")
##            test2=np.average(np.abs(test1/data))
##            print(test2)
#            return data
        size = np.shape(data)
        n = size[0]*size[1]
        m = np.sqrt(n)
        if self.useGPU:
            data_gpu =  self.thr.to_device(data.astype(np.complex64))
            program = rFFT(data_gpu)
            fftc = program.compile(self.thr, fast_math=True)
            fftc(data_gpu,data_gpu,inverse=1)
            if self.normalization:
                data_gpu *= m
            else:
                data_gpu *= n
            signal_fft = data_gpu.get()
        else:
            signal_fft = np.fft.ifft2(data)
            if self.normalization:
                signal_fft *= m
            else:
                signal_fft *= n
        return signal_fft
            

    def fftshift(self, data):
        if self.useGPU:
            data_gpu =  self.thr.to_device(data.astype(np.complex64))
            program = rFFTShift(data_gpu)
            fftc = program.compile(self.thr, fast_math=True)
            fftc(data_gpu,data_gpu)
            signal_fft = data_gpu.get()
        else:
            signal_fft = np.fft.fftshift(data)
        return signal_fft

