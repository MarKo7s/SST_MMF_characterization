from numpy import *
import numpy as np
from scipy.ndimage import center_of_mass
from numba import jit

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from threading import Semaphore, Thread

from pathlib import Path
import pickle

import time

class phase_retrival:
    
    def lockPhase(MTMs, jit = True):
        
        if jit == True:
            return lockPhase_numba(MTMs)
        else:
            return lockPhase_original(MTMs)
        
class frame_centering:
    
    @staticmethod
    def findcentercorrection(frame): #single frame support
        frame_dimensions = frame.shape
        frame_ideal_center = frame_dimensions[1]//2
        fc = center_of_mass(frame)
        center_start = fc
        YCorrection = int((frame_ideal_center - fc[0]))
        XCorrection = int((frame_ideal_center - fc[1]))
        frame_recentered = roll(frame, shift=(YCorrection,XCorrection),axis=(0,1))
        fc = center_of_mass(frame_recentered)
        Yerror = int(rint(frame_ideal_center - fc[0]))
        Xerror = int(rint(frame_ideal_center - fc[1]))
        frame_recentered = roll(frame_recentered, shift=(Yerror,Xerror), axis=(0,1))
        frame_center = center_of_mass(frame_recentered)
        center_after_correction = frame_center
        Yshift = YCorrection + Yerror
        Xshift = XCorrection + Xerror

        shift = [Yshift, Xshift]
        print("Centers:", center_start, "After recentering:", center_after_correction, "Correction needed:", shift)
        return frame_recentered, shift

    @staticmethod
    def applycenteroffset(frame, shift, frameaxes = (0,1,2), shiftaxes = (1,2)):
        frame_dimensions = frame.shape
        num_frames = 1
        if len(frame_dimensions) > 2:
            frame_width = frame_dimensions[frameaxes[2]] #axes[2] = 2, x axis
            frame_height = frame_dimensions[frameaxes[1]] #axes[1] - 1 , y axis
            num_frames = frame_dimensions[frameaxes[0]] # 
            
        else:
            frame_width = frame_dimensions[1]
            frame_height = frame_dimensions[0]
            frame = frame[None,...] #exten dimension to make it compatible with the multiple frame
        
        frame_recentered = roll(frame, shift=(shift[0],shift[1]), axis = shiftaxes)
        return(frame_recentered)
    

@jit(nopython = True, cache = True)
def lockPhase_numba(MTMs):
    # Adjust Horizontal first
    px = np.int32(sqrt(MTMs.shape[0]))
    outSpots_coefs = np.copy(MTMs) #This is as MTMs but with the phaseShift corrected
    phi = zeros((px,px), dtype = float64) #store the shift
    w = zeros_like(phi)
    px_up = arange(px//2 - 1, dtype = int32) + px//2 #This goes from 64 to 126
    px_down = flip(arange(px//2 + 1, dtype = int32))[0:-1]
    px_up_ = px_up + 1
    px_down_ = px_down - 1
    px_arr = concatenate((px_up,px_down))
    px_arr_ = concatenate((px_up_,px_down_))

    offset_marging = 0.0

    #pxIdxs = stack([px_arr,px_arr_],axis=-1) #nopython no-compatible
    pxIdxs = transpose(vstack((px_arr,px_arr_)))
    for i in pxIdxs:
        pxy = (px//2) #for the central line, travel horizontally -- fixed
        pxx = i[0] #reference pixel index
        pxx_ = i[1] #pixel to be locked to the reference
        
        Nidx = (px*(pxy)) + pxx #1D pixel index
        Nidx_ = (px*(pxy)) + pxx_ #1D pixel index
        
        ref = outSpots_coefs[Nidx,:] 
        target = outSpots_coefs[Nidx_,:] 
        
        Aphi = target * conj(ref)
        w[pxy,pxx_] = abs(sum(Aphi))**2
        Aphi_av = average(Aphi)
        Aphi_av = angle(Aphi_av)
        
        phi[pxy,pxx_] = Aphi_av
        
        outSpots_coefs[Nidx_,:] = outSpots_coefs[Nidx_,:] * exp(-1j*(Aphi_av))
    
    #Adjust rest of the pixels based on the previous adjusted line

    for pxx in range(px):
        for i in pxIdxs:
            pxy = i[0] #reference pixel index
            pxy_ = i[1] #pixel to be locked to the reference
            
            Nidx = (px*(pxy)) + pxx #1D pixel index
            Nidx_ = (px*(pxy_)) + pxx #1D pixel index
            
            ref = outSpots_coefs[Nidx,:] 
            target = outSpots_coefs[Nidx_,:] 
            
            Aphi = target * conj(ref)
            w[pxy_,pxx] = abs(sum(Aphi))**2
            Aphi_av = average(Aphi)
            Aphi_av = angle(Aphi_av)
            phi[pxy_,pxx] = Aphi_av

            outSpots_coefs[Nidx_,:] = outSpots_coefs[Nidx_,:] * exp(-1j*(Aphi_av)) 
            
    return outSpots_coefs, phi, w

def lockPhase_original(MTMs):
    # Adjust Horizontal first
    px = np.int32(sqrt(MTMs.shape[0]))
    outSpots_coefs = np.copy(MTMs) #This is as MTMs but with the phaseShift corrected
    phi = zeros((px,px), dtype = float64) #store the shift
    w = zeros_like(phi)
    px_up = arange(px//2 - 1, dtype = int32) + px//2 #This goes from 64 to 126
    px_down = flip(arange(px//2 + 1, dtype = int32))[0:-1]
    px_up_ = px_up + 1
    px_down_ = px_down - 1
    px_arr = concatenate((px_up,px_down))
    px_arr_ = concatenate((px_up_,px_down_))

    offset_marging = 0.0

    #pxIdxs = stack([px_arr,px_arr_],axis=-1) #nopython no-compatible
    pxIdxs = transpose(vstack((px_arr,px_arr_)))
    for i in pxIdxs:
        pxy = (px//2) #for the central line, travel horizontally -- fixed
        pxx = i[0] #reference pixel index
        pxx_ = i[1] #pixel to be locked to the reference
        
        Nidx = (px*(pxy)) + pxx #1D pixel index
        Nidx_ = (px*(pxy)) + pxx_ #1D pixel index
        
        ref = outSpots_coefs[Nidx,:] 
        target = outSpots_coefs[Nidx_,:] 
        
        #I shouldn't need to do this if power between pixel should mostly the same
        targetpow = sqrt(sum(abs(target)**2))
        refpow = sqrt(sum(abs(ref)**2))
        target_norm = target / targetpow
        ref_norm = ref/refpow
        
        Aphi = target * conj(ref)
        w[pxy,pxx_] = abs(sum(Aphi))**2
        Aphi_av = average(Aphi)
        Aphi_av = angle(Aphi_av)
        
        phi[pxy,pxx_] = Aphi_av
        
        outSpots_coefs[Nidx_,:] = outSpots_coefs[Nidx_,:] * exp(-1j*(Aphi_av))
    
    #Adjust rest of the pixels based on the previous adjusted line

    for pxx in range(px):
        for i in pxIdxs:
            pxy = i[0] #reference pixel index
            pxy_ = i[1] #pixel to be locked to the reference
            
            Nidx = (px*(pxy)) + pxx #1D pixel index
            Nidx_ = (px*(pxy_)) + pxx #1D pixel index
            
            ref = outSpots_coefs[Nidx,:] 
            target = outSpots_coefs[Nidx_,:] 
            
            targetpow = sqrt(sum(abs(target)**2))
            refpow = sqrt(sum(abs(ref)**2))
            target_norm = target / targetpow
            ref_norm = ref/refpow
            
            Aphi = target * conj(ref)
            w[pxy_,pxx] = abs(sum(Aphi))**2
            Aphi_av = average(Aphi)
            Aphi_av = angle(Aphi_av)
            phi[pxy_,pxx] = Aphi_av

            outSpots_coefs[Nidx_,:] = outSpots_coefs[Nidx_,:] * exp(-1j*(Aphi_av)) 
            
    return outSpots_coefs, phi, w


#! Sweep proccessing stuff:

class Sweep_Pulse_Specs():
    '''Class to store processing settings'''
    def __init__(self, centerWav, BW, N, pBW ):
        self.wc = centerWav
        self.wav_ini = BW[0]
        self.wav_end = BW[1]
        self.N = N #number of wavlengths (sampling number)
        if isinstance(pBW, (ndarray, list, tuple)):
            self.count = len(pBW)
            self.pBW = pBW
        else:
            #Hopefully is an integer
            self.count = 1
            self.pBW = []
            self.pBW.append(pBW)
        

class pulseFromSweep():
    '''Class that process sweep frames under parameters provided in Sweep_Pulse_specs'''
    def __init__(self, SwepAndPulseSpecs):
        self.SpecsWorkPack : Sweep_Pulse_Specs =  SwepAndPulseSpecs
        self.wc = self.SpecsWorkPack.wc
        self.wav_ini = self.SpecsWorkPack.wav_ini
        self.wav_end = self.SpecsWorkPack.wav_end
        self.N = self.SpecsWorkPack.N
        self.Bsamples = arange(self.N)
        self.BWarray = linspace(self.wav_ini, self.wav_end, self.N)
        self.BW_of_interest = self.SpecsWorkPack.pBW
        self.wav_count = self.SpecsWorkPack.count # Size of the array of BW provided
        self.wav_idx = [] #This is the importand parameters - it is used to integrate the pulse
        
        for BW_idx,BW_target in enumerate(self.BW_of_interest):
            self.setTargetBW(BW_target, BW_idx) #Fill wav_idx list with the arrays containing the integration ranges
        
    def setTargetBW(self, targetBW, BW_idx):
        wav_ini_sel = self.wc - targetBW/2
        wav_end_sel = self.wc + targetBW/2
        lowerBound = self.find_nearest_value(self.BWarray,wav_ini_sel)
        upperBound = self.find_nearest_value(self.BWarray,wav_end_sel) + 1 #+1 to compensate the indexing
        if lowerBound == upperBound:
            lowerBound = upperBound - 1
        tmp_wav_array = self.Bsamples[lowerBound:upperBound]
        samples = tmp_wav_array.shape[0]
        self.wav_idx.append(tmp_wav_array) #Samples of interest to form the pulse 
        resolution = targetBW / samples
        print(f'{BW_idx} - Target BW = {targetBW} - wavelength range {wav_ini_sel:0.2f} to {wav_end_sel:0.2f} with resolution {resolution * 1000:0.2f} pm with {samples} samples  ')
        
    def getPulse(self, sweepArray, BWIdx = 0): #Set the index by default to 0 just in case
        if self.wav_ini != None: 
            targetWavsFrames = average(sweepArray[self.wav_idx[BWIdx]], 0)
            return targetWavsFrames
        else:
            print('Target Bandwidth has not been set yet')
        
    @staticmethod
    def find_nearest_value(array, value):
        Vmax = array.max()
        Vmin = array.min()
        if value >= Vmax : value = Vmax
        if value <= Vmin : value = Vmin
        idx = where(array >= value)[0][0:2] #get the couple below the target
        errorIdx = abs(value - array[idx]).argmin()
        idx_nearest = idx[errorIdx]
        return idx_nearest

class AcqAndProccessingMemPlan():
    '''It creates a mem plan for the generated sweep pulse'''
    
    def __init__(self, framesize, num_projections, ProcessingBlockSize, AcquisitionBlockSize):
        self.frameSize = framesize
        self.totalProjections = num_projections
        self.ProcessingBlockSizeGB = ProcessingBlockSize
        self.AcquisitionBlockSizeGB = AcquisitionBlockSize
        
        self.savingSlicingRows, self.acquisitionSlicingProj, self.savingBlocks, self.acquisitionBlocks = self.memoryplan(framesize, num_projections, ProcessingBlockSize, AcquisitionBlockSize)
        
    @staticmethod
    def memoryplan(framesize, num_projections, ProcessingBlockSize, AcquisitionBlockSize):
        #Processing constrain:
        pixelsToProcess_bytes = ProcessingBlockSize * 1024**3 / (dtype(float32).itemsize * num_projections)
        totalpixels = framesize**2
        SavingBlocks = int(ceil(totalpixels / pixelsToProcess_bytes))
        #Acquisition constrain:
        pixelsToAcquire_bytes = AcquisitionBlockSize * 1024**3 / (totalpixels * dtype(float32).itemsize)
        AcquisitionBlocks = int(ceil(num_projections / pixelsToAcquire_bytes))
        #Create some indexes vectors:
        savingSlicing = linspace(0, totalpixels, SavingBlocks + 1, dtype = np.int32) // framesize # in rows --> total pixels is rows * framesize
        acquisitionSlicing = linspace(0, num_projections, AcquisitionBlocks + 1, dtype = np.int32) # in projections
        
        return(savingSlicing, acquisitionSlicing, SavingBlocks, AcquisitionBlocks)
    
    def updatePlan(self):
        self.savingSlicingRows, self.acquisitionSlicingProj, self.savingBlocks, self.acquisitionBlocks = self.memoryplan(self.frameSize, self.totalProjections, self.ProcessingBlockSizeGB, self.AcquisitionBlockSizeGB)


class FileBuffer():
    '''Class that load sweep frames'''
    def __init__(self, path, SweepFrameProcessingParameters, pol = 'H'):
        self.buffer = None #Buffer (numpy array)
        self.bufferIdx = 0
        self.frameIdx = 0
        self.frameRange = [0,-1]     
        self.pol = pol
                
        self.fileOpened = None
        self.folder_path = path
        #Open the specs file to get structure of the file
        path_to_specs = path + 'measurement_specs.pkl'
        with open(path_to_specs,'rb') as file:
            data = pickle.load(file)
            
        self.file_idx = data['FilesIdx']
        self.framesize = data['framesize']
        self.wavcount = data['wavelengthCount']  

        self.frameProcessor : pulseFromSweep = pulseFromSweep(SweepFrameProcessingParameters) 
        self.processingCount = self.frameProcessor.wav_count #Number of targeting pulses of different BW
        self.frameShape = (self.frameProcessor.N, self.framesize, self.framesize)
        self._stopAQ = False
        self._bufferRunning = False
        
        #Get frame
        self._waitForFrame = Semaphore(0)
        self._targetFrameIdx = 0
        
        
        
    def __repr__(self):
        return(f'Stats--> Range:{self.frameRange}, bufferIdx:{self.bufferIdx}, frameIdx:{self.frameIdx}')

        
    def fillBuffer_thread(self):
        self._bufferRunning = True
        for itIdx in range(self.bufferSize):

            try:
                loadedData = np.load(self.fileOpened).reshape(self.frameShape)
                # Loop generating pulse for one of each of the provided BW
                for BWidx in range(self.processingCount):
                    self.buffer[BWidx, itIdx] = self.frameProcessor.getPulse(loadedData, BWidx) #np.random.rand(self.framesize, self.framesize)# load and process for given BW
    
            except:
                print(f'Error loading--> BufferIdx:{itIdx} - FrameIdx:{self.frameIdx}')
                self.buffer[:,itIdx] = zeros((self.processingCount,self.framesize,self.framesize)) #Filled with zeros and notify
            finally:    
                self.bufferIdx = itIdx
                self.updateFrameIdx()
                if self._targetFrameIdx == self.frameIdx:
                    self._waitForFrame.release()
                                
                if self._stopAQ == True:
                    break
    
        self.closeFile()
        self._bufferRunning = False
        #self._waitForFrame.release()
    
    def openFile(self, index):
        self.fileOpened = open(Path(self.folder_path) / self.findFolder(index), 'rb')
        #Reset buffer and pointers fields
        self.frameRange = self.find_range(self.file_idx,index)
        self.bufferSize = self.frameRange[1] - self.frameRange[0]
        #print(f'Buffer size: { self.bufferSize}')
        self.buffer = zeros((self.processingCount, self.bufferSize, self.framesize, self.framesize), dtype = np.float32) #Allocate mem #? Index meaning: buffer[PulseBWidx, frameIdx, Y, X]
        self.updateFrameIdx()
        #Start the acquisition
        th = Thread(target = self.fillBuffer_thread )
        th.start()
        
    def closeFile(self):
        self.fileOpened.close()
    
    def findFolder(self, index):
        
        target_range = self.find_range(self.file_idx,index)
        target_folder = f'{target_range[0]}_{target_range[1]}/{self.pol}/file_{self.pol}.npy'
        
        return target_folder
        
    def GetFrame(self, index):
        
        """
        3 scenarios:a) Frame belong to a different file --> 1) Stop AQ 2) Close file 3) open new file 4)Start AQ
                    b) Frame belong to target file but still not in buffer --> wait and return the frame
                    c) Frame is ready --> return frame       
        """
        self._targetFrameIdx = index
        if self._isTargetFileAreadyOpen(index) == False:
            #print('Open File, and waiting for frame')
            self.openFile(index)   
            self._waitForFrame.acquire()
            #print(f'Target frame: {index}, No opened file')
        elif self._isFrameAlreadyInBuffer(index) == False:
            #print('Waiting for frame')
            self._waitForFrame.acquire()
            #print(f'Target frame: {index}, waiting')
        #print('Returning Frame')
        #print(self.frameIdx)
        return self.buffer[:, self.getBufferIdx(index), ...]
    
    #Save as getFrame but instead of returning writes the result into the provided mem.space
    def FrameToMemory(self, index, toArray):
        self._targetFrameIdx = index
        if self._isTargetFileAreadyOpen(index) == False:
            self.openFile(index)   
            self._waitForFrame.acquire()
        elif self._isFrameAlreadyInBuffer(index) == False:
            self._waitForFrame.acquire()
            
        toArray[...] = self.buffer[:, self.getBufferIdx(index), ...]
        
        return (1)
 
    def getBufferIdx(self, index):
        return index - self.frameRange[0] 
    
    def updateFrameIdx(self):
        self.frameIdx = self.bufferIdx + self.frameRange[0]
        
    def _isTargetFileAreadyOpen(self, index):
        
        return self.frameRange[0] <= index < self.frameRange[1] 
        
    def _isFrameAlreadyInBuffer(self, index):
        
        return self.frameIdx >= index
        

    @staticmethod
    def find_range(array, value):
        Vmax = array.max()
        Vmin = array.min()
        if value >= Vmax : value = Vmax
        if value <= Vmin : value = Vmin
        valueIdx = where(array == value)[0]
        if len(valueIdx)>0 :
            lowerValue = array[valueIdx][0]
            upperValue = array[valueIdx + 1][0]
        else:   
            idx_min = where(array <= value)[0][-1] #get last one
            idx_max = where(array >= value)[0][0] #get first one
            lowerValue = array[idx_min]
            upperValue = array[idx_max]
        ArrayRange = np.array([lowerValue, upperValue])
        return ArrayRange

class SweepDataToPulse():
    '''Loads, Process and saves the frames generating the pulse'''
    def __init__(self, path_load_sweep, path_dump_pulse, memplanObject, SweepPulseSpecsObject ):
        self.path_load_sweep = path_load_sweep
        
        self.SweepPulsesSpecs : Sweep_Pulse_Specs = SweepPulseSpecsObject
        self.path_dump_pulse = []
        self.BWcount = self.SweepPulsesSpecs.count
        for BWTarget in self.SweepPulsesSpecs.pBW:
            self.path_dump_pulse.append(self.createFolder(path_load_sweep, path_dump_pulse, self.SweepPulsesSpecs.wc, BWTarget)) # Autogenerate the folders if they do not exist where to store the measurements
        
        self.memplanObject : AcqAndProccessingMemPlan = memplanObject
        #First create new object to proces the sweep frames
        projectionBuffer_H : FileBuffer = FileBuffer(path = self.path_load_sweep, 
                                                        SweepFrameProcessingParameters = SweepPulseSpecsObject, pol = 'H')
        projectionBuffer_V : FileBuffer = FileBuffer(path = self.path_load_sweep, 
                                                        SweepFrameProcessingParameters = SweepPulseSpecsObject, pol = 'V')
        self.projectionBuffer = {'H': projectionBuffer_H, 'V': projectionBuffer_V}
        
        self.polChar = ['H', 'V']
        
    def GeneratePulseIntoDisk(self, from_block = 0, to_block = None, pol = 'HV', save = True):
        
        if (to_block == None or to_block > self.memplanObject.acquisitionBlocks or to_block == 0):
            to_block = self.memplanObject.acquisitionBlocks
            #print(f'Going to: {to_block}')
        
        for iterIdx in range(from_block, to_block):

            from_proj = self.memplanObject.acquisitionSlicingProj[iterIdx]
            to_proj = self.memplanObject.acquisitionSlicingProj[iterIdx+1]
            print(f'{iterIdx} - from {from_proj} to {to_proj} ')
            res = self.frameProcessing(from_proj, to_proj, pol= pol, save = save, iterIdx = iterIdx)
            print(f'{iterIdx} - {res}')
            
        print('Incoherent pulse simulation done')
        
        #Save specs file inside the folders:
        for BWIdx in range(self.BWcount):
            measurement_specs = {'projectioncount': self.memplanObject.totalProjections,
                    'framesize': self.memplanObject.frameSize, 'frameSlicing': self.memplanObject.savingSlicingRows, 
                    'AcquisitionSlicing': self.memplanObject.acquisitionSlicingProj, 'needreshape': True}

            path_specs = self.path_dump_pulse[BWIdx] + f'measurement_specs.pkl' #Make H pol as the master path

            with open(path_specs,'wb') as file:
                pickle.dump(measurement_specs, file)
                    
        
    def frameProcessing(self, from_proj, to_proj, pol , save = True, iterIdx = 0):
        
        projSize = to_proj - from_proj
        batch = {}
        batch.clear()
        
        #Create the buffer
        for polIdx in pol:
            batch[polIdx] = zeros((self.BWcount, projSize, self.memplanObject.frameSize, self.memplanObject.frameSize), dtype = float32)
        
        futures = []
        r = 0
        for fIdxm,projIdx in enumerate(range(from_proj, to_proj)):
            futures *= 0
            futures = []
            #print(f'Projection = {projIdx}')
            with ThreadPoolExecutor() as executor:
                for polIdx in pol:
                    #print(f'Pol = {polIdx} - Projection = {projIdx}')
                    futures.append(executor.submit(self.projectionBuffer[polIdx].FrameToMemory, projIdx, batch[polIdx][:,fIdxm,...])) #Copies everything to the buffer
                    #wait until done
                for fut in as_completed(futures):
                    r+=fut.result()
        #Saving according memplan               
        if save == True:
            #!Loop to save each pulse from here
            print(f'Saving Total {r} frames out {projSize} projections with {pol} for {self.BWcount} pulses')
            
            for BWIdx in range(self.BWcount):
                futures*=0
                r = 0
                with ThreadPoolExecutor() as executor:
                    for sliceIdx in range(self.memplanObject.savingBlocks):
                        from_row = self.memplanObject.savingSlicingRows[sliceIdx] 
                        to_row =  self.memplanObject.savingSlicingRows[sliceIdx + 1] 
                        for polIdx in pol:
                            frame = batch[polIdx][BWIdx,:,from_row:to_row,:]
                            futures.append(executor.submit(self.dumpData, frame.ravel(), self.path_dump_pulse[BWIdx], polIdx, sliceIdx, iterIdx))
                            
                for fut in as_completed(futures):
                    r += fut.result()
                #print(f'Pulse : {BWIdx} done')
            return f'{pol} - {self.BWcount} Pulses --> saved'
        else:        
            return (batch, f'{pol}- {r} projections - {self.BWcount} Pulses --> Done')   
        
    @staticmethod
    def createFolder( path_data, path_to_save, wc_label, BWlabel):
        myPath = Path(path_data)
        folder_name = myPath.stem
        #path_to_pulse = f'{path_to_save}{folder_name}_{wc_label}nm_BW_{BWlabel}nm' 
        path_to_pulse = f'{path_to_save}{folder_name}'
        path_to_pulse = Path(path_to_pulse)
        if path_to_pulse.exists() == False : path_to_pulse.mkdir() #create a new directory to store the data
        path_to_pulse = str(path_to_pulse) + f'\\BW_{BWlabel}nm'
        path_to_pulse = Path(path_to_pulse)
        if path_to_pulse.exists() == False : path_to_pulse.mkdir() #create a new directory to store the data
        path_to_pulse = str(path_to_pulse) + '\\'
        return(path_to_pulse)
    
    @staticmethod
    def testStatic(a,b):
        print(a,b)
        
    @staticmethod
    def dumpData(array, folder_path, polIdx, sliceIdx, iterationIdx):
        #finish the path
        path = folder_path + polIdx + f'_{iterationIdx}_{sliceIdx}.npy'
        #save
        with open(path,'wb') as file:
            np.save(file, array )
            
        return 1



#! Debugging:
def test(a):
    futures = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, a):
            futures.append(executor.submit(doTask, i))
            
    return(a)
def doTask(b):
    return(b)

if __name__ == '__main__':
    path = 'Z:\\stokes\\sweep_intensity_measurements\\221014_5_mode_groups_118_wav\\'
    wc = 1300
    wav_ini = 1295
    wav_end = 1305
    N = 118
    pulseGenerator = pulseFromSweep(wc, [wav_ini,wav_end], N)
    Bsel = 2
    pulseGenerator.setTargetBW(Bsel)
    
    p = []
    t = time.time()
    for idx, polIdx in enumerate(['H','V']):
        p.append( Process(target=frameProcessing, args=((0, 100, polIdx, 118, (118,200,200), path, pulseGenerator ))))
        p[idx].start()
        print('process', idx)
    
    p[0].join()
    p[1].join()
    t2 = time.time()
    print('Done', t2-t)