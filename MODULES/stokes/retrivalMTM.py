import sys
import os
import pathlib
from pathlib import Path
p = pathlib.Path(__file__).parent.parent
path_to_module = p

from fibremodes import ModesGen as mg
from fibremodes.ModesGen import overlaps
from stokes.stokestomography import *
from stokes.processingTools import frame_centering, phase_retrival

import pickle

from IPython.display import clear_output



class Stokes_Tomography_MTM_retrival:
    def __init__(self, StokesTomographySetUp, eigenvector_extractionCount = 3, MFD_retrival = 17.4, forcePSD = False, GPU = True):
        self.GellMan = StokesTomographySetUp['Gellman']
        self.stokesWeights = StokesTomographySetUp['stokeWeights']
        self.pol = ['H','V']
        self.modegroupcountMAX = StokesTomographySetUp['modegroup']
        self.modecount = int(0.5 * self.modegroupcountMAX * (self.modegroupcountMAX + 1))
        
        self.PSD_flag = forcePSD
        self.GPU_flag = GPU
        
        self.MFD_retrival = MFD_retrival
        
        self.eigenvector_extractionCount = eigenvector_extractionCount
        #Measurement related stuff:
        self.framesize = None
        self.projectioncount = None
        self.frameSlicing = None
        self.AcquisitionSlicing = None
        self.frameslicingSize = None
        self.AcquisitionSlicingSize = None

    
        self.spot_array_coefs = None
        self.MTM = {}
        self.stokes_vector_results = {}
        
        #Saving:
        #self.saving_path = None
        #self.datafilename = None
        
    
    def __generate_spot_array_matrix(self):
        w0 = self.MFD_retrival
        mfd = w0*2
        ng = self.modegroupcountMAX
        px = self.framesize
        Apx = 1

        LGbases = mg.LGmodes(mfd,ng,px,Apx, generateModes = True, wholeSet = True, engine ='GPU', multicore = True) 
        mm_gra = LGbases.LGmodesArray__
        modesCount = mm_gra.shape[0]

        LG2D = reshape(mm_gra,(mm_gra.shape[0],-1)) #2D LG to use for overlaps
        LG2D = transpose((LG2D))
        spot_array = eye(px**2,px**2)
        spot_array = reshape(spot_array,(px**2,px,px))
        self.spot_array_coefs = overlaps.Modal_decomposition((LG2D), spot_array) # THIS IS THE MATRIX TRANSFORMATION TO APPLY
        

    def setFrameSize(self, px):
        if self.framesize != px:
            print('Calculating spot array matrix ...')
            self.framesize = px       
            self.__generate_spot_array_matrix()

    #First thing to call when a new processing routine is called
    def load_measurement_specs(self, path):
        path_to_file = path + 'measurement_specs.pkl'
        with open(path_to_file,'rb') as file:
            data = pickle.load(file)
        framesize = data['framesize']
        self.setFrameSize(framesize) #It will update the spot array if framesize changed
        self.projectioncount = data['projectioncount']
        self.frameSlicing =  data['frameSlicing']
        self.AcquisitionSlicing = data['AcquisitionSlicing']
        self.frameslicingSize = len(self.frameSlicing) - 1
        self.AcquisitionSlicingSize = len(self.AcquisitionSlicing) - 1
        
    def process_measurement(self, path_to_measurement, label): 
        
        #being lazy
        def doOverlap2(a,b):
            a = a / sqrt(sum(abs(a)**2))
            b = b / sqrt(sum(abs(b)**2))
            
            ov = average(a*conj(b))
            return ov 
        
        self.load_measurement_specs(path_to_measurement)  
        
        #temporal varibales
        number_eigenVectors = self.eigenvector_extractionCount
        framesize = self.framesize
        GellMan = self.GellMan
        modecount = self.modecount
        pol = self.pol
        
        #allocate mem:
        #eigenvectors
        MTMs_H = zeros((number_eigenVectors, framesize**2,int32(ceil(sqrt(GellMan.shape[0])))), complex64)
        MTMs_V = zeros((number_eigenVectors, framesize**2,int32(ceil(sqrt(GellMan.shape[0])))), complex64)
        MTMs = {'H': MTMs_H, 'V': MTMs_V}
        #eigenvalues
        EigenVal_H = zeros((framesize**2,modecount*len(pol)), float32)
        EigenVal_V = zeros((framesize**2,modecount*len(pol)), float32)
        EigenVal = {'H':EigenVal_H, 'V':EigenVal_V}
        
        #Stokes Vector Calculation:
        for polIdx in pol:
            for sliceIdx in range(len(self.frameSlicing)-1):
                from_row = self.frameSlicing[sliceIdx] #// framesize
                to_row = self.frameSlicing[sliceIdx + 1] #// framesize
                from_px = from_row * framesize
                to_px = to_row * framesize
                totrow = to_row - from_row
                frame_tmp = zeros((self.projectioncount, totrow, framesize))
                #print(from_px, to_px)
                
                for itIdx in range(len(self.AcquisitionSlicing)-1):
                    file_name = path_to_measurement + f'{polIdx}_{itIdx}_{sliceIdx}.npy'
                    from_proj = self.AcquisitionSlicing[itIdx]
                    to_proj = self.AcquisitionSlicing[itIdx + 1]
                    
                    totproj = to_proj - from_proj
                    
                    rec_frame_shape = (totproj, totrow, framesize ) 
                    #Extraction:
                    #print(f'Extracting {sliceIdx} containing {totrow} rows of pixels. Projections: {from_proj}/{to_proj}')
                    frame_tmp[from_proj:to_proj, ...] = load(file_name).reshape(rec_frame_shape)
                    #frames[polIdx][from_proj:to_proj,from_row:to_row,:] = frame_tmp[from_proj:to_proj, ...]
                    
                
                clear_output( wait=True)   
                print(f'{polIdx} - Processing {sliceIdx} from row {from_row} to row {to_row}')
                #processing:
                Sn, densityMatrix, eigenValues, eigenVectors, S0 = StokesVectorCalc(GellMan, self.stokesWeights, (frame_tmp + 1e-23 ),
                                                                                    ForcePSD = self.PSD_flag, GPU = self.GPU_flag)
                for eigenVIdx in range(number_eigenVectors):
                    targetEigenVector = eigenVIdx
                    MTMs[polIdx][targetEigenVector, from_px:to_px,...] = eigenVectors[:,:,targetEigenVector] * sqrt(S0[:,None]) 
                    EigenVal[polIdx][from_px:to_px,...] = eigenValues[:,:]
            
            minEignVal = amin(EigenVal[polIdx], axis = 1)
            EigenVal[polIdx] = EigenVal[polIdx] / sum(EigenVal[polIdx], 1)[:,None]
            
        clear_output(wait = True)
            
        #centering:
        MTMs_3D_H = reshape( MTMs['H'], (number_eigenVectors,framesize, framesize, modecount * len(pol)) )
        MTMs_3D_V = reshape( MTMs['V'], (number_eigenVectors, framesize, framesize, modecount * len(pol)) )   
        frameTotalH_centered, Hshift = frame_centering.findcentercorrection( sum(abs(MTMs_3D_H[0])**2, 2) )  
        frameTotalV_centered, Vshift = frame_centering.findcentercorrection( sum(abs(MTMs_3D_V[0])**2, 2) )
        #correct:
        MTMs_3D_H_centered = zeros_like(MTMs_3D_H)
        MTMs_3D_V_centered = zeros_like(MTMs_3D_V)

        for i in range(number_eigenVectors):
            frameTotalH_centered, Hshift = frame_centering.findcentercorrection( sum(abs(MTMs_3D_H[i])**2, 2) )  
            frameTotalV_centered, Vshift = frame_centering.findcentercorrection( sum(abs(MTMs_3D_V[i])**2, 2) )
            
            MTMs_3D_H_centered[i] = frame_centering.applycenteroffset(MTMs_3D_H[i], shift = Hshift, frameaxes = (1,2,0), shiftaxes = (0,1))
            MTMs_3D_V_centered[i] = frame_centering.applycenteroffset(MTMs_3D_V[i], shift = Vshift, frameaxes = (1,2,0), shiftaxes = (0,1))  
            
        MTMs_2D_centered_H = reshape(MTMs_3D_H_centered, (number_eigenVectors, framesize**2, modecount * len(pol)))
        MTMs_2D_centered_V = reshape(MTMs_3D_V_centered, (number_eigenVectors, framesize**2, modecount * len(pol)))
        
        clear_output(wait = True)
        
        MTMs = {'H': MTMs_2D_centered_H, 'V': MTMs_2D_centered_V}
        
        self.stokes_vector_results[label] = {'MTMs': MTMs, 'EigenValues': EigenVal}
        
        #Phase locking:
        
        outSpots_coefs = {'H': zeros_like(MTMs['H']), 'V': zeros_like(MTMs['V'])}
        Aphi = {'H': None, 'V': None}
        w  = { 'H': None, 'V': None}
        
        for i in range(number_eigenVectors):
            for polidx in pol:
                print(f'Correcting: {polidx}')
                outSpots_coefs[polidx][i], Aphi[polidx], w[polidx] = phase_retrival.lockPhase(MTMs[polidx][i], jit = True)    
                    
        print('Phase locking done')   
        clear_output(wait = True) 
        #MTM retrival
        
        MTM_retrived_pol = {'H': None,'V': None}
        MTM_retrived = zeros((number_eigenVectors, modecount * len(pol), modecount * len(pol)), complex64)

        for i in range(number_eigenVectors):
            for polidx in pol:
                MTM_retrived_pol[polidx] = matmul(transpose(outSpots_coefs[polidx][i]), self.spot_array_coefs)

            MTM_retrived[i,:,::2] = MTM_retrived_pol['H']
            LockPols = angle(doOverlap2(MTM_retrived_pol['H'], MTM_retrived_pol['V']))
            MTM_retrived[i,:,1::2] = MTM_retrived_pol['V'] * exp(-1j*LockPols)

        self.MTM[label] = MTM_retrived 
        
        print(f'MTM {label} retrieved')       
        

    #plotting:
    def showMTMSVD(self, label):
        for i in range(self.eigenvector_extractionCount):
            u, s, vh = linalg.svd(self.MTM[label][i], full_matrices=True)
            #plot(10*log10(U_retrived.diagonal() / (U_retrived.diagonal()).max()), label = 'diagonal power')
            plot(10*log10(s/s.max()),label = f'svd {i}')
            title('svd')
            ylabel('dB')
            #ylim([-10,1])
            legend()
            eigChan = 10*log10(s/s.max())
            supported_eigChannels = where(eigChan>-10)[0]
            #print('Channels', supported_eigChannels)
            s_norm = s /s.max()
            MDL = 10*log10(s_norm.min())
            IL = s.shape[0] - (sum(s/s.max()))
            print(f'MDL {i}', MDL)
            print(f'IL {i}', IL)
    
    def showMTMSVD_interact(self, label, ax):
        for i in range(self.eigenvector_extractionCount):
            u, s, vh = linalg.svd(self.MTM[label][i], full_matrices=True)
            eigChan = 10*log10(s/s.max())
            supported_eigChannels = where(eigChan>-10)[0]
            s_norm = s /s.max()
            MDL = 10*log10(s_norm.min())
            IL = s.shape[0] - (sum(s/s.max()))
            
            ax.plot(10*log10(s/s.max()),label = f'eigenstate {i} - MDL {MDL:.2f} - IL {IL:.2f}')
            ax.set_ylabel('attenuation (dB)')
            ax.set_xlabel('sigular values')
            ax.legend()
        
            
    def showMTMsSVD(self, label):
        for i in range(self.eigenvector_extractionCount):
            MTMs_interlev = vstack((self.stokes_vector_results[label]['MTMs']['H'][i], self.stokes_vector_results[label]['MTMs']['V'][i]))
            u, s, vh = linalg.svd(MTMs_interlev, full_matrices=False)
            plot(10*log10(s/s.max()),label = f' eigenState {i}')
            title(f'svd')
            ylabel('dB')
            #xlim([0,14])
            legend()

            s_norm = s /s.max()
            print(f'MDL eigState {i}', 10*log10(s_norm[-1])) #Obviously only 14 channels are supported in one pol
            print(f'IL  eigState {i}', s.shape[0] - (sum(s/s.max())))
            

    def save_MTM_into_Disk(self, path):
        with open(path,'wb') as file:
            pickle.dump(self.MTM, file)
            
    def save_StokesVectors_into_Disk(self, path):
        with open(path,'wb') as file:
            pickle.dump(self.stokes_vector_results, file)
            