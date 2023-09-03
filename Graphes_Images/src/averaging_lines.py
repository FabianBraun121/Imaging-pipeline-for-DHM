# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:29:57 2023

@author: SWW-Bc20
"""
import os
os.chdir(r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\src\spatial_averaging')
import binkoala
import numpy as np
from  pyKoalaRemote import client
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import minimize, Bounds
from scipy import ndimage
from skimage.registration import phase_cross_correlation

#%%
class LineDrawer:
    def __init__(self, img):
        self.img = img
        self.line = None
        self.points = []
        self.line_points = []  # Store points on the line
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.enter_pressed = False  # Added to track Enter key press

    def on_click(self, event):
        # Check for valid click (within image, left mouse button)
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.line is None:  # if there is no line, create a line
            self.line, = self.ax.plot([np.round(event.ydata,0)], [np.round(event.xdata,0)], 'r')  # plot the point
            self.points.append((int(np.round(event.ydata,0)), int(np.round(event.xdata,0))))
        else:  # if there is a line, dynamically update the line
            self.line.set_xdata([self.points[-1][1], np.round(event.xdata,0)])
            self.line.set_ydata([self.points[-1][0], np.round(event.ydata,0)])
            self.points.append((int(np.round(event.ydata,0)), int(np.round(event.xdata,0))))
            self.fig.canvas.draw()


    def bresenham_line(self, pt1, pt2):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end

        points_input: (12,37) (61,18)
        points_output: [(12,37),(13,36),...,(61,18)]
        """
        # Setup initial conditions
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()

        return points

    def wait_for_enter(self):
        def on_key(event):
            if event.key == 'enter':
                self.enter_pressed = True

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        while not self.enter_pressed:
            plt.pause(0.1)
        plt.close()

    def show_image_and_draw(self):
        # Display image
        self.ax.imshow(self.img, cmap='gray')
        plt.show()

def draw_on_image(img):
    line_drawer = LineDrawer(img)
    line_drawer.show_image_and_draw()
    line_drawer.wait_for_enter()

    if len(line_drawer.points) < 2:
        print("Not enough points clicked. Please click two points on the image.")
        return []

    line_drawer.line_points = line_drawer.bresenham_line(line_drawer.points[-2], line_drawer.points[-1])
    return line_drawer.line_points

class Hologram:
    def __init__(self, fname):
        self.fname = fname
        self.reconstrution_distance_left = -3
        self.reconstrution_distance_right = -1
        self.focus = None # Focus distance
        self.focus_method = None # evaluation function
        self.focus_score = None # score of evaluatino function at the Focus point (minimum)
        self.optimizing_method = None # function descent method
        self.plane_basis_vectors = None # normally polynomial basis vectors 
        self.plane_fit_order = None # also named degree
        self.X_plane = None # since the evaluation points, aka imagepoints stay constant X for linear regression is always the same
        self.X_plane_pseudoinverse = None # together with X_plane here to calculate linear regression, less computational cost since constant for all images
        self.cache_path = 'C:\\Users\\SWW-Bc20\\Documents\\GitHub\\Imaging-pipeline-for-DHM\\data\\__file'
        self.koala_host = None # client with which interaction with koala takes place
        self.cplx_image = None # as soon as the focus point is found this function is evaluated
    
    def calculate_focus(self, koala_host, focus_method='std_sobel_squared', optimizing_method= 'Powell', tolerance=None,
                        x0=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):

        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetUnwrap2DState(True)
        self.X_plane = self._X_plane()
        self.X_plane_pseudoinverse = self._X_plane_pseudoinverse()
        if x0 == None:
            x0 = (self.reconstrution_distance_left+self.reconstrution_distance_right)/2
        bounds = Bounds(lb=self.reconstrution_distance_left, ub=self.reconstrution_distance_right)
        res = minimize(self._evaluate_reconstruction_distance, [x0], method=self.optimizing_method, bounds=bounds)
        self.focus = res.x[0]
        self.focus_score = res.fun
        self.cplx_image = self._cplx_image(use_amp=use_amp)
            
    def _cplx_image(self, use_amp=True):
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SetRecDistCM(self.focus)
        self.koala_host.OnDistanceChange()
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        ph, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        ph = self._subtract_plane(ph)
        if use_amp:
            self.koala_host.SaveImageFloatToFile(2,self.cache_path+'_amp.bin',True)
            amp, __header = binkoala.read_mat_bin(self.cache_path+'_amp.bin')
            cplx = amp*np.exp(complex(0.,1.)*ph)
        else:
            cplx = np.exp(complex(0.,1.)*ph)
        return cplx
    
    def _evaluate_reconstruction_distance(self, reconstruction_distance):

        self.koala_host.SetRecDistCM(reconstruction_distance[0])
        self.koala_host.OnDistanceChange()
        # saves the phase of the hologram
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        image_values, __header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        image_values = self._subtract_plane(image_values)
        if self.focus_method == 'Max_std_of_phase_squard':
            # saves the phase of the hologram
            image_values *= image_values
            #returns the negativ since most optimizing function look for the Minimum
            return -np.std(image_values)
        elif self.focus_method == 'std_sobel_squared':
            return -self._evaluate_std_sobel_squared(image_values)
        else:
            print("Method ", self.focus_method, " to find the focus point is not implemented.")
    
    def _evaluate_std_sobel_squared(self, gray_image):
        # Calculate gradient magnitude using Sobel filter
        grad_x = scipy.ndimage.sobel(gray_image, axis=0)
        grad_y = scipy.ndimage.sobel(gray_image, axis=1)
        # Calculate std squared sobel sharpness score
        return np.std(grad_x ** 2 + grad_y ** 2)
        
    def get_cplx_image(self):
        return self.cplx_image.copy()
    
    def header(self):
        self.koala_host.LoadHolo(self.fname,1)
        self.koala_host.SaveImageFloatToFile(4,self.cache_path+'_ph.bin',True)
        __ph, header = binkoala.read_mat_bin(self.cache_path+'_ph.bin')
        return header
    
    def _subtract_plane(self, field):
        theta = np.dot(self.X_plane_pseudoinverse, field.reshape(-1))
        plane = np.dot(self.X_plane, theta).reshape(field.shape[0], field.shape[1])
        return field-plane
    
    def _X_plane(self):
        if self.plane_basis_vectors == "Polynomial":
            ## Relevel all images with a plane before averaging. This removes most errors with missalignment due to DHM errors
            ## Stolen from https://stackoverflow.com/questions/35005386/fitting-a-plane-to-a-2d-array
            X1, X2 = np.mgrid[:800, :800]
            X = np.hstack((X1.reshape(-1,1) , X2.reshape(-1,1)))
            return PolynomialFeatures(degree=self.plane_fit_order, include_bias=True).fit_transform(X)
        else:
            print(self.plane_basis_vectors,  ' is not implemented')
    
    def _X_plane_pseudoinverse(self):
        return np.dot( np.linalg.inv(np.dot(self.X_plane.transpose(), self.X_plane)), self.X_plane.transpose())

class SpatialPhaseAveraging:
    def __init__(self, loc_dir, timestep, koala_host, focus_method='std_sobel_squared', optimizing_method= 'Powell',
                 tolerance=None, plane_basis_vectors='Polynomial', plane_fit_order=5, use_amp=True):
        self.loc_dir = loc_dir
        self.timestep = timestep
        self.koala_host = koala_host
        self.focus_method = focus_method
        self.optimizing_method = optimizing_method
        self.tolerance = tolerance
        self.plane_basis_vectors = plane_basis_vectors
        self.plane_fit_order = plane_fit_order
        self.use_amp = use_amp
        self.x0_guess = None
        self.pos_list = [ f.name for f in os.scandir(loc_dir) if f.is_dir()]
        self.num_pos = len(self.pos_list)
        self.focus_score_list = []
        self.cplx_images_list = []
        self.holo_list = self._generate_holo_list()
        self.holo_list_in_use = self.holo_list
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    
    def _background(self):
        background = self.holo_list_in_use[0].get_cplx_image()
        background = self._subtract_bacterias(background)
        for i in range(1, len(self.holo_list_in_use)):
            cplx_image = self.holo_list_in_use[i].get_cplx_image()
            cplx_image = self._subtract_phase_offset(cplx_image, background)
            cplx_image = self._subtract_bacterias(cplx_image)
            background += cplx_image
        return background/len(self.holo_list_in_use)
    
    def _cplx_avg(self):
        cplx_avg = self.holo_list_in_use[0].get_cplx_image()
        cplx_avg /= self.background
        self.cplx_images_list.append(cplx_avg)
        for i in range(1, len(self.holo_list_in_use)):
            cplx_image = self.holo_list_in_use[i].get_cplx_image()
            cplx_image /= self.background
            cplx_image = self._shift_image(cplx_avg, cplx_image)
            cplx_image = self._subtract_phase_offset(cplx_image, cplx_avg)
            cplx_avg += cplx_image
            self.cplx_images_list.append(cplx_image)
        return cplx_avg/len(self.holo_list_in_use)
    
    def _generate_holo_list(self):
        holo_list = []
        for pos in self.pos_list:
            fname = self.loc_dir + os.sep + pos + os.sep + "Holograms" + os.sep + str(self.timestep).zfill(5) + "_holo.tif"
            holo = Hologram(fname)
            holo.calculate_focus(self.koala_host, focus_method=self.focus_method, optimizing_method=self.optimizing_method, tolerance=self.tolerance,
                                 x0=self.x0_guess, plane_basis_vectors=self.plane_basis_vectors, plane_fit_order=self.plane_fit_order)
            # first guess is the focus point of the last image
            self.x0_guess = holo.focus
            self.focus_score_list.append(holo.focus_score)
            holo_list.append(holo)
        return holo_list
    
    def get_amp_avg(self):
        return np.absolute(self.cplx_avg)
    
    def get_cplx_avg(self):
        return self.cplx_avg.copy()
    
    def get_mass_avg(self):
        ph = np.angle(self.cplx_avg)
        cut_off = 0.15
        return np.sum(ph[cut_off<ph])
    
    def get_phase_avg(self):
        return np.angle(self.cplx_avg)
    
    def restrict_holo_use(self, holo_used):
        self.holo_list_in_use = [self.holo_list[i] for i in holo_used]
        self.num_pos = len(holo_used)
        self.background = self._background()
        self.cplx_avg = self._cplx_avg()
    
    def _shift_image(self, reference_image, moving_image):
        shift_measured, error, diffphase = phase_cross_correlation(np.angle(reference_image), np.angle(moving_image), upsample_factor=10, normalization=None)
        shiftVector = (shift_measured[0],shift_measured[1])
        #interpolation to apply the computed shift (has to be performed on float array)
        real = ndimage.shift(np.real(moving_image), shift=shiftVector, mode='wrap')
        imaginary = ndimage.shift(np.imag(moving_image), shift=shiftVector, mode='wrap')
        return real+complex(0.,1.)*imaginary
    
    def _subtract_bacterias(self, cplx_image):
        # subtracts pixel  that are far away from the mean and replaces them with the mean of the image
        # cut off value is determined by hand and has to be reestimated for different use cases
        cut_off = 0.15
        ph = np.angle(cplx_image)
        ph[cut_off<ph] = np.mean(ph[ph<cut_off])
        return np.absolute(cplx_image)*np.exp(1j*ph)
    
    def _subtract_phase_offset(self, new, avg):
        z= np.angle(np.multiply(new,np.conj(avg))) #phase differenc between actual phase and avg_cplx phase
        #measure offset using the mode of the histogram, instead of mean,better for noisy images (rough sample)
        hist = np.histogram(z,bins=1000,range=(np.min(z),np.max(z)))
        index = np.argmax(hist[0])
        offset_value = hist[1][index]
        new *= np.exp(-offset_value*complex(0.,1.))#compensate the offset for the new wavefront
        return new
    

def connect_to_remote_koala(ConfigNumber):
    # Define KoalaRemoteClient host
    host = client.pyKoalaRemoteClient()
    #Ask IP address
    IP = 'localhost'
    # Log on Koala - default to admin/admin combo
    host.Connect(IP)
    host.Login('admin')
    # Open config
    host.OpenConfig(ConfigNumber)
    host.OpenPhaseWin()
    host.OpenIntensityWin()
    host.OpenHoloWin()
    return host

#%%
base_path = r'C:\Users\SWW-Bc20\Documents\GitHub\Imaging-pipeline-for-DHM\Graphes_Images'
data_path = base_path + r'\data\E10_20230216\2023-02-16 12-14-58\00001'
save_path = base_path + r'\Graphes_Images\averaging_lines'


#%%
########################## start Koala and define functions ##########################
ConfigNumber = 221
host = connect_to_remote_koala(ConfigNumber)
spa = SpatialPhaseAveraging(data_path, 0, host)
#%%
images = np.array([np.angle(cplx_img) for cplx_img in spa.cplx_images_list])
avg_image = spa.get_phase_avg()

np.save(save_path + os.sep + 'images', images)
np.save(save_path + os.sep + 'avg_image', avg_image)
#%%
points = draw_on_image(images[0])
#%%
np.save(save_path + os.sep + 'points', np.array(points))
#%%
images = np.load(save_path + os.sep + 'images.npy')
avg_image = np.load(save_path + os.sep + 'avg_image.npy')
points = np.load(save_path + os.sep + 'points.npy')
points = [tuple(points[i]) for i in range(points.shape[0])]

#%%

for image in images:
    line = np.array([image[p[::-1]] for p in points])
    plt.plot(line, 'b')
plt.plot(np.array([avg_image[p[::-1]] for p in points]), 'r', label='averaged images')
plt.xlabel('pixel')
plt.ylabel('phase [rad]')




#%%
plt.figure(figsize=(4.5,4))
plt.plot(np.array([images[0][p] for p in points])*794/(2*np.pi), 'r', label='pixel values')
plt.plot([2,28],[25,25], label='avg. value')
plt.legend(fontsize=12)
plt.xlabel('Pixels red line', fontsize=12)
plt.ylabel('Optical path difference [nm]', fontsize=12)

#%%
img = images[0]

plt.imshow(images[0])
