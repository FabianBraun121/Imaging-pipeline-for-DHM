# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:46:20 2014

Methods useful in the different class

@author: tcolomb
"""
import sys
if sys.version_info[0:2] == (2,7):
    python_vers = "2.7"
if sys.version_info[0:2] > (3,5):
    python_vers = "3.x"
import os
import csv
import numpy as np
import string
from itertools import combinations
from itertools import product
from scipy.ndimage.interpolation import zoom
# if python_vers == "2.7":
#     from guidata.qt.QtGui import QMessageBox, QFileDialog
#     from Tkinter import *
# if python_vers == "3.x":
#     from plotpy.qt.QtGui import QMessageBox, QFileDialog
#     from tkinter import *
from PyQt5.QtWidgets import QFileDialog
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
def getEntry(requestMessage,boolTextOrNumber,defaultText, hideText):
    msgBox=takeInput(requestMessage,boolTextOrNumber,defaultText, hideText)
    msgBox.waitForInput()
    if boolTextOrNumber: #True=text, False=Number
        return msgBox.getString()
    else:     
        return int(float(msgBox.getString()))

def files_number(path,extension):
    list_dir = []
    count = 0
    try:
        list_dir = os.listdir(path)
        for file in list_dir:
            if file.endswith(extension):
                count += 1
    except:
        count=-1
      
    return count
    
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def parentDirectory(directory):
    return os.path.abspath(os.path.join(directory, os.pardir))
def write_txt_file(fname, arrayparams):
    f = open(fname, 'w')
    for params in arrayparams:
        f.writelines(str(params))
        f.writelines('\n')
    f.close()
    
def read_txt_file(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        arrayparams = []
        for row in reader:
            line = row[0]
            arrayparams.append((float)(line))
    return arrayparams
       
def Choose_Destination_File(directory):
    fname = QFileDialog.getSaveFileName(None, "Save params", directory)
    if python_vers == "3.x":
        fname = fname[0]
    return fname
    

def Open_Directory(directory, message):
    #print(directory)
    fname = QFileDialog.getExistingDirectory(None, message, directory, QFileDialog.ShowDirsOnly)
#    if python_vers == "3.x":
#        fname = fname[0]
    return fname

def Open_Txt_File(directory):
    fname = QFileDialog.getOpenFileName(None, "Open params", directory)
    if python_vers == "3.x":
        fname = fname[0]
    return fname
    
    
def OpenFile(selected_filter, directory=None):
    fname = QFileDialog.getOpenFileName(None, "File Dialog", directory, selected_filter)
    if python_vers == "3.x":
        fname = fname[0]
    return fname

def fileExists(fname):
    return os.path.exists(fname)
    
def sum_string_from_list(string_list):
    name = ""
    for string_value in string_list:
        name += string_value+" "
    return name
    
def useless_combination(comb):
    """
    Define if a combintation is useless
    """
    test1 = len(comb) > len(set(comb))  and len(set(comb)) != 1 #suppress combination where there are two times the same value [1,1,2] but keep [1,1] (set([1,1]={1}))
    test2 = comb != sorted(comb) #suppress combination that are not from smallest value to largest value
    return test1 or test2

def find_holo_format(holopath):
    """
    Find the hologram format to be opened in the directory
    """
    if fileExists(holopath+'.tif'):
        return holopath+'.tif'
    elif fileExists(holopath+'.png'):
        return holopath+'.png'
    elif fileExists(holopath+'.bmp'):
        return holopath+'.bmp'
    else:
        return None
def Define_holoname(directory):
    holoname = "Hologram"
    try:
        list_dir = os.listdir(directory)
        continue_to_search = True
        k=0
        while continue_to_search and k<len(list_dir):
            file = list_dir[k]
            if file.endswith('.tif') or file.endswith('.png') or file.endswith('bmp'):
                file = file[:-4]
                if file.endswith('_+45') or file.endswith('_-45'):
                    file = file[:-4]
                file = file[::-1]
                try:
                    (int)(file[:2])
                    file = file[2:]
                    holoname = file[::-1]
                    continue_to_search = False
                except:
                    try:
                        (int)(file[:1])
                        file = file[1:]
                        holoname = file[::-1]
                        continue_to_search = False
                    
                    except:
                        continue_to_search = True
                
                
                continue_to_search = False
            k += 1
    except:
        print('directory does not exist: '+directory)
    return holoname
    
def powers(n):
    """
    convert the contrastLambda in list of reconstructionId to process
    """
    list_to_process = [2**p for p,v in enumerate(bin(n)[:1:-1]) if int(v)]
    list_to_process = np.log2(list_to_process)
    list_to_process = [int(i) for i in list_to_process]
    return list_to_process

def construct_choicecontrast_list(db):
    listSourceID = []
    choicecontrast_list_name = []
    choicecontrast_list_value = []
    alpha = string.ascii_lowercase
    for it, sourceIdLensParams in enumerate(db.listLensParams.ListLensParams):
        sourceId = sourceIdLensParams.sourceId
        name = str(sourceId)
        count = len([i for i in listSourceID if i==sourceId])
        if count != 0:
            name += '_'+alpha[count]
        listSourceID.append(sourceId)
        choicecontrast_list_name.append(name)
        choicecontrast_list_value.append(2**it)
    all_name_combinations = sum([list(map(list, combinations(choicecontrast_list_name, i))) for i in range(len(choicecontrast_list_name) + 1)], [])
    all_value_combinations = sum([list(map(list, combinations(choicecontrast_list_value, i))) for i in range(len(choicecontrast_list_value) + 1)], [])
    all_sourceId_combinations =  sum([list(map(list, combinations(listSourceID, i))) for i in range(len(listSourceID) + 1)], [])
    choicecontrast_list = []
    for i, comb in enumerate(all_value_combinations):
        add = len(comb) != 0
        if add:
            useless_combinations = useless_combination(all_sourceId_combinations[i])
            if useless_combinations:
                add = False
        if add:
            choicecontrast_list.append((sum(comb), 'Lambda '+sum_string_from_list(all_name_combinations[i])))
    choicecontrast_list.append((-1, "VertScan"))
    choicecontrast_list.append((-2, "Reflecto"))
    return choicecontrast_list
def construct_contrast_method_list(list_to_process, *args):
    contrast_method_list=[]
    if len(args) == 0:
        contrast_method_list.append((-1,"single"))
        contrast_method_list.append((6, "Mapping on VertScan"))
        contrast_method_default = 0
    else:
        if len(list_to_process) == 1:
            if list_to_process[0] != -1 and list_to_process[0] != -2:
                contrast_method_list.append((-1,"single"))
                contrast_method_list.append((5, "Seg. mapp"))
                contrast_method_list.append((6, "Mapping on VertScan"))
                contrast_method_default = -1
            if list_to_process[0] == -1:
                contrast_method_list.append((-2,"Only VertScan"))
                contrast_method_default = -2
            if list_to_process[0] == -2:
                contrast_method_list.append((-3,"Reflecto"))
                contrast_method_default = -3
        elif len(list_to_process) == 2:
            srceId_1 = args[0].get_wavelength_m_from_id(list_to_process[0])[0]
            srceId_2 = args[0].get_wavelength_m_from_id(list_to_process[1])[0]
            if srceId_1 == srceId_2:
                contrast_method_default= 4
                contrast_method_list.append((3,"complex add"))
                contrast_method_list.append((4,"difference"))
            else:
                contrast_method_default = 0
                contrast_method_list.append((0,"long synth"))
                contrast_method_list.append((1,"mapping"))
                contrast_method_list.append((2,"short synth"))
                contrast_method_list.append((3,"complex add"))
                contrast_method_list.append((4,"difference"))
                contrast_method_list.append((5, "Seg. mapp"))
                contrast_method_list.append((6, "Mapping on VertScan"))
        elif len(list_to_process) > 2:
            contrast_method_default = 1
            contrast_method_list.append((1,"mapping"))
            contrast_method_list.append((5, "Seg. mapp"))
            contrast_method_list.append((3,"complex add"))
            contrast_method_list.append((6, "Mapping on VertScan"))
    return contrast_method_list, contrast_method_default
    
def construct_lambdaChoice_list(db):
    lambdaChoice_list = []
    for it, wavelength in enumerate(db.wavelengths_nm):
        name = str(it+1)
        lambdaChoice_list.append((it, 'lambda '+name))
    lambdaChoice_default = 0
    return lambdaChoice_list, lambdaChoice_default
    
def construct_holorefChoice_list(db):
    listSourceID = []
    holorefsPath_default = []
    holorefChoice_list = []
    for it, sourceIdLensParams in enumerate(db.listLensParams.ListLensParams):
        holorefPath = sourceIdLensParams.LensParameter.holorefpath
        sourceId = sourceIdLensParams.sourceId
        name = str(sourceId)
        count = len([i for i in listSourceID if i==sourceId])
        if count != 0:
            name += '_'+str(count+1)
        listSourceID.append(sourceId)
        holorefsPath_default.append(holorefPath)
        holorefChoice_list.append((it, name))
    return holorefChoice_list, holorefsPath_default
    
def construct_lensChoice_list(db):
    listSourceID = []
    lensChoice_list = []
    for it, sourceIdLensParams in enumerate(db.listLensParams.ListLensParams):
        sourceId = sourceIdLensParams.sourceId
        if sourceId != 1: #lambda 1 have default value for the lens
            name = str(sourceId)
            count = len([i for i in listSourceID if i==sourceId])
            if count != 0:
                name += '_'+str(count+1)
            listSourceID.append(sourceId)
            lensChoice_list.append((it, name))
    if len(lensChoice_list) == 0:
        show = False
        lensChoice_list.append((0,"no lens"))
        lensChoice_default = 0
    else:
        show = True
        lensChoice_default = 1
    return lensChoice_list, lensChoice_default, show
    
def construct_viblensChoice_list(db):
    listSourceID = []
    viblensChoice_list = []
    for it, sourceIdLensParams in enumerate(db.listLensParams.ListLensParams):
        sourceId = sourceIdLensParams.sourceId
        name = str(sourceId)
        count = len([i for i in listSourceID if i==sourceId])
        if count != 0:
            name += '_'+str(count+1)
        listSourceID.append(sourceId)
        viblensChoice_list.append((it, name))
    viblensChoice_default = 0
    return viblensChoice_list, viblensChoice_default
    
def gss(a, b, function, *args):
    """
    Golden Search Method
    """
    maincode = args[0]
    c = b -  (b - a) / maincode.phi
    d = a + (b - a) / maincode.phi
    while abs(c - d) > maincode.gss_tolerance:
        shift_measured, f_c, diffphase = function(c,*args)
        shift_measured, f_d, diffphase = function(d,*args)
        if f_c < f_d:
            b = d
        else:
            a = c
        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / maincode.phi
        d = a + (b - a) / maincode.phi
    return (b + a) / 2.   
 
def get_closest_elements(arr_1):
    closest = sorted(product(arr_1, arr_1), key=lambda t: abs(t[0]-t[1]))
    criteria = 0
    k = 0
    while criteria == 0:
        criteria = closest[k][0]-closest[k][1]
        out = closest[k]
        k += 1
    sorted(out)    
    return out
    
def ShowWarningMessage(title, message):
    QMessageBox.warning(None, title, message, QMessageBox.Ok)
    
def SaveCancelMessage(title, message):
    res = QMessageBox.warning(None, title, message, QMessageBox.Save, QMessageBox.Cancel)
    return res == QMessageBox.Save

def resize_wavefront(w, scaling):
    r = np.real(w)
    img = np.imag(w)
    r = zoom(r, 1/scaling)
    img = zoom(img, 1/scaling)
    return r + complex(0.,1.)*img

def opencsvfile(fname, n_column):
    data = []
    for k in range(n_column):
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            arrayparams = []
            l = 0
            for row in reader:
                if l>0:
                    line = row[k]
                    arrayparams.append((float)(line))
                l += 1
        data.append(arrayparams)
    return data

def read_timestamps(fname):
    data = []
    with open(fname, 'r') as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            data.append([(int)(row[0]),row[1],row[2],(float)(row[3])])
    data = np.array(data)
    return data

def read_timestamps_scan(fname):
    data = []
    with open(fname, 'r') as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            data.append([(int)(row[0]),row[1],row[2],(float)(row[3]),(float)(row[4]),(float)(row[5]),(float)(row[6])])
    data = np.array(data)
    return data[0]
    