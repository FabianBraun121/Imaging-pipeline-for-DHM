# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:25:52 2022

@author: Henshaw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:06:28 2015

@author: tcolomb
"""
import sys
if sys.version_info[0:2] == (2,7):
    python_vers = "2.7"
if sys.version_info[0:2] > (3,5):
    python_vers = "3.x"


# if python_vers == "2.7":
#     from guidata.qt.QtGui import QFileDialog

# if python_vers == "3.x":
#     from plotpy.qt.QtGui import QFileDialog


import os
# import cv2
import numpy as np
import PIL.Image
import shutil
import csv
from tkinter import Tk,filedialog

class Sequence:
    def __init__(self,inputdirectoryPath,outputdirectoryPath,first_holo_seq,last_holo_seq, holosize=1024):
        self.inputdirectoryPath=inputdirectoryPath
        self.outputdirectoryPath=outputdirectoryPath
        self.first_holo_seq=first_holo_seq
        self.last_holo_seq=last_holo_seq
        self.holoWidth=holosize
        self.holoHeight=holosize
        self.bpp=None
        self.framesNumber=None
    def CreateOutputDirectory(self):
        if not os.path.exists(self.outputdirectoryPath+'\\Holograms'):
            os.makedirs(self.outputdirectoryPath+'\\Holograms')
    def DefineSeqFormat(self):
        holodir=self.inputdirectoryPath+'\\Holograms'
        if os.path.isfile(holodir+'\\holo.avi'):
            self.seqFormat=0
            # cap=cv2.VideoCapture(holodir+'\\holo.avi')
            # self.holoWidth=cap.get(3)
            # self.holoHeight=cap.get(4)
            # self.framesNumber=cap.get(7)
            # self.bpp=8
            # cap.release()
            
        elif os.path.isfile(holodir+'\\holo.raw'):
            self.seqFormat=1
            kbin_header_dtype = np.dtype([
             ("width", "i4"), 
             ("height", "i4"),
             ("bpp", "i4"),
             ("frames_number", "i4")])
            f = open(holodir+'\\holo.raw', 'rb')
            kbin_header = np.fromfile(f, dtype=kbin_header_dtype, count=1) 
            self.holoWidth = (int)(kbin_header['width'])
            self.holoHeight = (int)(kbin_header['height'])
            self.bpp=kbin_header['bpp']
            self.framesNumber=(int)(kbin_header['frames_number'])
        else:
            self.seqFormat=2
            self.framesNumber=self.frames_Number()
            self.bpp=8
            
    def frames_Number(self):
        list_dir = []
        count = 0
        try:
            list_dir = os.listdir(self.inputdirectoryPath+'\\Holograms')
            for file in list_dir:
                if file.endswith('.tif'):
                    count += 1
        except:
            count=-1
          
        return count
    def save_xth_holo(self,xth_input,xth_output):
        inputdirectory=self.inputdirectoryPath+'\\Holograms'
        saveholopath=self.outputdirectoryPath+'\\Holograms\\'+str(xth_output).zfill(5)+'_holo.tif'
        # if self.seqFormat==0:
            # cap=cv2.VideoCapture(inputdirectory+'\\holo.avi')
            #ret, frame =cap.read()
            # frameCounter=0
            # play=True
            # while(play):
            #     ret,frame=cap.read()
            #     if frame is not None:
            #         if frameCounter==xth_input:
            #             img=PIL.Image.fromarray(frame)
            #             img.save(saveholopath)
            #             play=False
            #         frameCounter+=1
            #     else:
            #         play=False
            # cap.release()
            
        if self.seqFormat==1:
            f = open(inputdirectory+'\\holo.raw', 'rb')
            beginat=xth_input*(self.holoHeight*self.holoWidth)
            size=self.holoHeight*self.holoWidth
            f.seek(beginat+16,1)
            tmp = np.fromfile(f, dtype='uint8',count=size)
            f.close
            Z=np.reshape(tmp,(self.holoHeight,self.holoWidth))          
            img=PIL.Image.fromarray(Z)
            img.save(saveholopath)
        
        elif self.seqFormat==2:
            inputholopath=inputdirectory+'\\'+str(xth_input).zfill(5)+'_holo.tif'
            #outputholopath=self.outputdirectory+'\\Holograms\\'+str(xth_input).zfill(5)+'_holo.tif'
            #self.frames_number=self.frames_Number(self.inputdirectory+'\\Holograms','.tif')
            #self.bpp=8
            shutil.copyfile(inputholopath,saveholopath)
    def saveCropSequence(self):
        xth_output=0
        for xth_input in range(self.first_holo_seq,self.last_holo_seq+1):
            if xth_input<self.framesNumber:
                self.save_xth_holo(xth_input,xth_output)
            else:
                print('lastholo is larger than frame number')
            xth_output+=1
    def WriteTimestamps(self):
        LinestoWrite=[]
        inputTimestamps=self.inputdirectoryPath+'\\timestamps.txt'
        with open(inputTimestamps,'r') as f:
            reader=csv.reader(f,delimiter=' ')
            p=0
            for row in reader:
                if p==self.first_holo_seq:
                    offset_time=float(row[3])
                if p>=self.first_holo_seq and p<=self.last_holo_seq:
                    holoNumber=str((int)(row[0])-self.first_holo_seq).zfill(5)
                    time=str((float)(row[3])-offset_time)
                    row[0]=holoNumber
                    row[3]=time                   
                    LinestoWrite.append(row[0]+' '+row[1]+' '+row[2]+' '+row[3]+'\n')
                p+=1
        f.close()
        outputTimestamps=self.outputdirectoryPath+'/timestamps.txt'
        f=open(outputTimestamps,'w')
        for line in LinestoWrite:
            f.writelines(line)
        f.close()

holosize = 1024        
default_dir= r'C:/Users/Admin/Downloads/'
#cropSequences=[(0,10),(30,50),(398,399)] #[(firstholo1,lastholo1),(firstholo2,lastholo2),...]
cropSequences=[(0,100)]
inputsequence=filedialog.askdirectory();
outputsequences=filedialog.askdirectory();
# inputsequence=str(QFileDialog.getExistingDirectory(None, "Choose Input Sequence",default_dir))
# outputsequences=str(QFileDialog.getExistingDirectory(None, "Choose Output Directory for sequences",default_dir))
name_input_sequence=os.path.basename(os.path.normpath(inputsequence))
for k in range(np.shape(cropSequences)[0]):
    outputsequence=outputsequences+'\\'+name_input_sequence+'_'+str(cropSequences[k][0]).zfill(5)+'_TO_'+str(cropSequences[k][1]).zfill(5)
    cropsequence = Sequence(inputsequence,outputsequence,cropSequences[k][0],cropSequences[k][1],holosize)
    cropsequence.CreateOutputDirectory()
    cropsequence.DefineSeqFormat()
    cropsequence.saveCropSequence()
    cropsequence.WriteTimestamps()