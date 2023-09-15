import sys
import os
import pathlib
from pathlib import Path
p = pathlib.Path(__file__).parent.parent
path_to_module = p
sys.path.append(str(path_to_module))#Routines uses:

from pylab import *
import pickle
import h5py
import hdf5storage

from fibremodes import ModesGen as mg
from fibremodes.ModesGen import overlaps
from custom_plotting import complexToRGB
from digholography.plugins import holography2
import re #regex for creating string

from stokes.processingTools import  frame_centering

from holography import Zernikes
from scipy.optimize import minimize, Bounds


#! Overlaps:
def normPow (a):
    return a / sqrt(sum(abs(a)**2))

def doOverlap(a,b):
    ov = abs( sum( a * conj(b) ) )**2 / (sum(abs(a)**2) * sum(abs(b)**2)) * 100
    return ov

def doOverlap_vec(reference, targets):
    ov = abs( sum( reference[None,...] * conj(targets), axis = (1,2) ) )**2 / (sum(abs(reference)**2) * sum(abs(targets)**2, axis = (1,2))) * 100
    return ov
    
def MaxOverlap(a,b):
    N = 5
    w_ov = zeros((2*N,2*N))
    for idx,i in enumerate(range(-N,N)):
        for jdx, j in enumerate(range(-N,N)):
            shifted_cam = frame_centering.applycenteroffset(b, shift = [i,j])[0]
            w = doOverlap(a, shifted_cam)

            w_ov[idx,jdx] = w
    
    max_ov = w_ov.max()
    pos = arange(-N,N)
    shift = where(w_ov == max_ov)
    shiftX = pos[shift[0]][0]
    shiftY = pos[shift[1]][0]
    shift = [shiftX, shiftY]
    
    return w_ov.max(), shift

#! Phase offsets
def findPhaseOffset(target,ref):
    target = target / sqrt(sum(abs(target)**2))
    ref = ref / sqrt(sum(abs(ref)**2))
    
    ov = target * conj(ref)
    Aphi_av = angle(average(ov))
    
    return(Aphi_av)

def complexDiff(a,b):
    a = normPow(a)
    b = normPow(b)
    
    c = a * conj(b)
    
    return c

#! Aberration correction :

class aberration_correction():
    def __init__(self, framesize, reference, target):
        self.framesize = framesize
        self.reference = reference
        self.target = target
        self.z = Zernikes(max_zernike_radial_number = 2 , mask_size_x_in_pixels = framesize, mask_size_y_in_pixels = framesize, 
                          aperture_diameter_in_m = framesize , SLM_pixel_size_in_um=1, load_modelab = False)
        
    def opti_function(self, x, *args):
        self.z.clean()
        self.z.clean()
        self.z.tilt_x_h = x[0]
        self.z.tilt_y_h = x[1]
        self.z.w_h[1] = x[2]
        
        self.z.make_zernike_fields()
        
        applyTilt = args[1] * exp(1j* angle(self.z.field_h))
        
        error = 100 - doOverlap(args[0], applyTilt)
        
        return error
    
    def find_aberrations(self, x0 = [0,0,0], max_value = 5):
        x0 = array(x0)
        limitsBound = Bounds(lb=-max_value, ub = max_value)

        res = minimize(self.opti_function, x0, args = (self.reference, self.target),
                    method='Nelder-Mead', bounds = limitsBound, options={'disp': True,'fatol':0.001, 'adaptive' : True})
        
        self.tiltX_opti = res.x[0]
        self.tiltY_opti = res.x[1]
        self.defocus_opti = res.x[2]
            
#! This auxiliar functions are used to generate the phase zernike phase and lambda dependency
def calcTilts(lambda0, lambdas, tiltX0, titltY0):
    ThetaUnits = pi/180
    thetaX0 = ThetaUnits * tiltX0
    thetaY0 = ThetaUnits * titltY0
    thetaX_lambdas = arcsin ( (lambda0 / lambdas) * sin(thetaX0))
    thetaY_lambdas = arcsin ( (lambda0 / lambdas) * sin(thetaY0))
    tiltX_lambdas = thetaX_lambdas / ThetaUnits
    tiltY_lambdas = thetaY_lambdas / ThetaUnits
    return tiltX_lambdas, tiltY_lambdas

def calcDioptres(lambda0, lambdas, dioptre0):
    return ((lambdas / lambda0) * dioptre0)

def getAberration(tiltX0, tiltY0, defocus, zernikes ):
    zernikes.clean()
    zernikes.tilt_x_h = tiltX0
    zernikes.tilt_y_h = tiltY0
    zernikes.w_h[1] = defocus # defocus
    zernikes.make_zernike_fields()
    aberration = angle(zernikes.field_h)
    return(aberration)           
           
#! Signal processing:

#! Analysis and plotting:

class incoherent_fields_analysis():
    def __init__(self, lambda_c, lambda_ini, lambda_end, BW, BW_MTM, TARGET_FIELDS, REFERENCE, aberrationCorrention = True):
        self.MTMCount = len(BW_MTM)      
        FieldsDimension = len(TARGET_FIELDS)
        self.FIELDS = TARGET_FIELDS
        self.FIELDS_ORIGINAL = TARGET_FIELDS #Save a copy of the input fields
        self.REFERENCE = REFERENCE
        self.N = TARGET_FIELDS[0].shape[0] # This should be the wavelenght axis
        self.px = TARGET_FIELDS[0].shape[1]
        
        self.lambda0 = lambda_c
        self.lambda_end = lambda_end
        self.lambda_ini = lambda_ini
        self.wav_axis = linspace(self.lambda_ini, self.lambda_end, self.N, endpoint = True)
        self.BW_MTM = BW_MTM
        self.BW = BW
        
        self.FIELD_OPTI = TARGET_FIELDS[0][self.N//2,...] # Use the most coherent measure patterns (ideally MTM_BW = 0 nm) at center wavelenght 
        self.phase_correction = ones((self.N,self.px,self.px), complex64)
        
        if aberrationCorrention == True:
            self.calc_aberration_correction() #This fill the phase correction
            self.apply_aberration_correction() #This applies the phase correction to all the fields
        
        #Analysis results:
        field_overlaps = zeros((self.MTMCount, self.N), float64)
        intensity_overlaps = zeros((self.MTMCount, self.N), float64)
        self.list_stats_name = ['field' , 'intensity']
        self.overlaps = {'field': field_overlaps, 'intensity': intensity_overlaps}
        self.BW_array_N = arange(1, self.N//2) # All avaliable BW we could integrate all overlaps
        field_BW_compount_overlap = zeros((self.BW_array_N.shape[0], self.MTMCount))
        intensity_BW_compount_overlap = zeros_like(field_BW_compount_overlap)
        self.BW_compount_overlap =  {'field': field_BW_compount_overlap, 'intensity': intensity_BW_compount_overlap}
        self.BW_axis = linspace(0,self.BW, self.BW_array_N.shape[0])

        
    def calc_aberration_correction(self):
        self.aberrations = aberration_correction(framesize = self.px, reference = self.REFERENCE, target = self.FIELD_OPTI)
        self.aberrations.find_aberrations()
        self.TiltX_wav, self.TiltY_wav = calcTilts(self.lambda0, self.wav_axis, self.aberrations.tiltX_opti, self.aberrations.tiltY_opti)
        self.Defocus_wav = calcDioptres(self.lambda0, self.wav_axis, self.aberrations.defocus_opti)
            
        for WAV_IDX in range(self.N):
            self.phase_correction[WAV_IDX] = exp(1j * getAberration(self.TiltX_wav[WAV_IDX], self.TiltY_wav[WAV_IDX], self.Defocus_wav[WAV_IDX], zernikes = self.aberrations.z)) 
            
    def apply_aberration_correction(self):
        #FIX THE FIELDS:
        #The phase correction should be commom to all the fields for all the MTM
        for MTM_IDX in range(self.MTMCount):
            self.FIELDS[MTM_IDX] = self.FIELDS_ORIGINAL[MTM_IDX] * self.phase_correction # Modify always the input original fields
        
            
    def run_analysis(self):
        
        for MTM_IDX in range(self.MTMCount):
            #Basic overlap: reference fields VS measure FIELDS(lambda) for each MTM --> OVERLAP[MTM_IDX][WAV_IDX] form
            self.overlaps['field'][MTM_IDX,...] = doOverlap_vec(reference = self.REFERENCE, targets = self.FIELDS[MTM_IDX])
            self.overlaps['intensity'][MTM_IDX,...] = doOverlap_vec(reference = abs(self.REFERENCE)**2, targets = abs(self.FIELDS[MTM_IDX])**2 )
            
        #Overlap integrated across all possible BW
        
        for idx, BW_idx in enumerate(self.BW_array_N):
            for name in self.list_stats_name:
                self.BW_compount_overlap[name][idx,:] = average(self.overlaps[name][:, self.N//2 - BW_idx : self.N//2 + BW_idx], 1)
    
    def plot_analysis(self):
        fig, ax = subplots(2,2,figsize = (15,10))
                        
        for MTM_IDX in range(self.MTMCount):
            for nameIDX, name in enumerate(self.list_stats_name):
                ax[nameIDX,0].plot(self.wav_axis, self.overlaps[name][MTM_IDX], label= f'MTM: {self.BW_MTM[MTM_IDX]} nm') 
                ax[nameIDX,1].plot(self.BW_axis, self.BW_compount_overlap[name][:,MTM_IDX], label= f'MTM: {self.BW_MTM[MTM_IDX]}')
        
        #totally overkilled
        for nameIDX, name in enumerate(self.list_stats_name):
            self.get_analysis_stats(name = name)
            cutoff = self.cut_off_3dB
            cutoff_array_field = ones(self.N) * cutoff
            cutoff_array_intensity = ones(self.BW_axis.shape[0]) * cutoff
            ax[nameIDX, 0].plot(self.wav_axis, cutoff_array_field, '--', color = 'black', label = 'Best -3dB')
            ax[nameIDX, 1].plot(self.BW_axis, cutoff_array_intensity, '--', color = 'black', label = 'Best -3dB')
                
        for nameIDX, name in enumerate(self.list_stats_name):         
            ax[nameIDX,0].set_xlabel('Wavelength (nm)')
            ax[nameIDX,1].set_xlabel('Bandwidth (nm)')
            ax[nameIDX,0].set_ylabel(f'{name} overlap (%)')
            ax[nameIDX,0].legend()
            ax[nameIDX,1].legend()
    
    #Get peaks where max overlaps etc etc. Intereting stuff to quickly check
    def get_analysis_stats(self, name = 'field'):
        max_overlaps_wav = self.wav_axis[where(self.overlaps[name] == self.overlaps[name].max(axis = 1)[:,None])[1]]
        max_values_overlaps = self.overlaps[name].max(axis = 1)
        
        absolute_peak = max_values_overlaps.max() # Reference
        cut_off_3dB = absolute_peak/2
        self.cut_off_3dB = cut_off_3dB
        
        def find_closest(input_array, value):
            error = abs(input_array - value)
            positions = error.argmin(axis = 1)
            return positions
        
        lowerValue = (self.N//2 - 1) - find_closest(self.overlaps[name][:,0:self.N//2], cut_off_3dB) #position from the center towards the left
        upperValue = find_closest(self.overlaps[name][:,(self.N//2-1)::], cut_off_3dB) # position from the center towards the right
        BW3dB = (self.BW / self.N ) * (upperValue + lowerValue) #Sum both regions and multuply for the waveleght units
                
        statistics = {'peaks': max_values_overlaps, 'lambdas': max_overlaps_wav, 'BW3dB': BW3dB}
        return statistics

    def calculate_MTM_intensity_patterns(self):
        fields_array = array(self.FIELDS)
        def get_N_BW(BW_target, N, BW):
            AN = BW/N
            N_target = BW_target/AN
            if N_target == 0:
                N_target = 1
            return(N_target)
        
        self.INCOHERENT = {}
        for idx, BWIdx in enumerate(self.BW_MTM):
            N_integration = int(ceil(get_N_BW(BWIdx,self.N,self.BW)/2))
            incoh = average(abs(fields_array[:, self.N//2 - N_integration:self.N//2 + N_integration,...])**2, (1))
            self.INCOHERENT[str(BWIdx)] = incoh
    
    def plot_incoherent_intensity_patterns(self):
        counter = self.MTMCount
        fig,ax = subplots(counter, counter, figsize = (10,10))
        
        for i,target_BW in enumerate(self.INCOHERENT.keys()):
            for j in range(counter):
                ax[i,j].imshow(self.INCOHERENT[target_BW][j], cmap = 'gray')
                ax[i,j].set_axis_off()
                
              
    def plot_FIELD(self, MTM_IDX, WAV_IDX):
        
        fig,ax = subplots(1,3,figsize = (20,10))
        
        MTM_idx = MTM_IDX
        wav_idx = WAV_IDX
        
        reference = self.REFERENCE
        target = self.FIELDS[MTM_idx][wav_idx,...]

        ax[0].imshow(complexToRGB(reference))
        LockPols = findPhaseOffset(target = target ,ref = reference)
        adjusted = target * exp(-1j*LockPols)
        print('Overlap: ', doOverlap(reference, adjusted))
        ax[1].imshow(complexToRGB(adjusted))
        ax[2].imshow(abs(target)**2, cmap = 'gray')
                
        
class impulse_response():
    def __init__(self, wav_ini, wav_end, N):
        self.wav_ini = wav_ini * 1e-9
        self.wav_end = wav_end * 1e-9
        c = 299792458
        fend = c / self.wav_ini
        fini = c / self.wav_end
        fBW = fend - fini
        print('frequency Badnwdith: ' , fBW * 1e-12, ' THz')
        Af = fBW / N
        self.At  = 1 / fBW
        print('Temporal resolution:', self.At * 1e12, ' ps')
        tmax = N//2 * self.At
        dmax = c * tmax
        Ad = c*self.At
        print('tmax: ', tmax * 1e12, ' ps - dmax: ',dmax * 1e3, ' mm' )
        self.taxis = arange(-N//2, N//2) * self.At * 1e12
        self.daxis = arange(-N//2, N//2) * Ad * 1e3
        
        self.impulse_response = None
        self.gaus_filter = None
         
    def filter_modes_temporal(self, coefs_in, sigma):
        coefs_FFT = fft((coefs_in), axis = 1) # FFT wavlength
        coefs_t_total = sum(abs(coefs_FFT)**2, 2)
        self.impulse_response = coefs_t_total
        
        find_peaks = (where(coefs_t_total  == coefs_t_total.max(axis  = 1 )[:,None] )[1])
        wav_size = coefs_in.shape[1]
        offset = find_peaks - wav_size//2
        self.gaus_filter = zeros((coefs_in.shape[0], wav_size))
        
        for IDX, offSET in enumerate(offset):
            self.gaus_filter[IDX] =  self.Gaussian1D(N = wav_size , sigma = sigma, offset = offSET, norm = False)
            
        coefs_t_filtered = coefs_FFT * self.gaus_filter[:,:, None]
        coefs_in_filtered = ifft((coefs_t_filtered), axis = 1)
        
        return(coefs_in_filtered)
     
    def plot_impulse_response(self, idx):
        figure()
        plot(self.daxis, self.impulse_response[idx]/ self.impulse_response[idx].max())
        plot(self.daxis, self.gaus_filter[idx])
        xlabel('distance (mm)')       
      
    @staticmethod
    def Gaussian1D(N, sigma = 1, offset = 0, norm = False):
        x = arange(-N//2, N//2)
        g = exp( (-1/2) * ( (x - offset) / sigma)**2 )
        if norm == True:
            g = g / sqrt(sum(abs(g)**2))
        else:
            g /= g.max()
        return(g)
            
    
#imports:
#Incoherent measurements

def import_incoherent_fields(path, modesCount, modesCrop, modeBasis, results_count = None):
    
    path_to_stokes_MTM = path
    stokes_MTM_results = sorted(os.listdir(path_to_stokes_MTM), key=len)

    counter = 0

    #fields
    fields_H = []
    fields_V = []
    #mode coefs
    coefs_H = []
    coefs_V = []
    #fields modes
    fields_H_modes = []
    fields_V_modes = []

    for fileIdx, filename in enumerate(stokes_MTM_results):
        if filename.__contains__('.mat'):
            counter += 1
            data =  hdf5storage.loadmat(path_to_stokes_MTM+filename)
            field = data['fieldR'] + (1j*data['fieldI'])
            field_H = field[0::2,...]
            field_V = field[1::2,...]
            field_H_coefs = data['coefs'][:, 0:modesCount//2]
            field_V_coefs = data['coefs'][:, modesCount//2:-1]
            field_H_coefs = field_H_coefs[:, 0:modesCrop]
            field_V_coefs = field_V_coefs[:, 0:modesCrop]
            
            coefs_H.append(field_H_coefs)
            coefs_V.append(field_V_coefs)
            
            fields_H_modes.append((rot90(fliplr(overlaps.Modal_reconstruction(modeBasis, field_H_coefs)), k = -1, axes= (1,2))))
            fields_V_modes.append((rot90(fliplr(overlaps.Modal_reconstruction(modeBasis, field_V_coefs)), k = -1, axes= (1,2))))
            
            print(f'{fileIdx} - {filename}') 
            fields_H.append(field_H)
            fields_V.append(field_V)
            if results_count != None:
                if(counter == results_count): break  
                
    return fields_H_modes, fields_V_modes, coefs_H, coefs_V
        