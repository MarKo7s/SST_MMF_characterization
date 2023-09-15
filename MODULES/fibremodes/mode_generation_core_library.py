# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:26:39 2020

Set of static functions to compute LG (maybe one day also HG) modes

@author: Marcos
"""

import numpy as np
from pylab import *
import numexpr as ne
import scipy
import scipy.special as sp
#simport numba_special
import time


#high speed packages that maybe the user does not have
try:
    import cupy as cp
except ModuleNotFoundError:
   print("Cupy module couldn't be imported")
try:
    from numba import njit
    
except ModuleNotFoundError:
    print("numba module couldn't be imported")
try:
   from ipyparallel import Client
except ModuleNotFoundError:
     print("Ipyparallel couldn't be imported")
   

   
############################## AUXILIAR FUNCTIONS #############################
##############  ##############  ##############  ##############  ##############    
   
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class times:
    def tic():
        global start_time
        start_time = time.time()
        return(start_time)

    def toc():
        elapsed_time = time.time() - start_time
        print("Elapsed time = ",elapsed_time)
        return(elapsed_time)

############################## AUXILIAR FUNCTIONS END #########################
##############  ##############  ##############  ##############  ###############

########################## MODES GENERATION FUNCTIONS #########################
##############  ##############  ##############  ##############  ############## 

def graded_index_fiber_coefs(xx):
    
    """ 
    Compute coef of the LG modes
    
    """
    group = xx
    if group%2 != 0:
        pp = np.arange( group//2 +1 ) + 1
    else:
        pp = np.arange(group//2) + 1   
    c = 1
    m = []
    k = 0
    n = []
    for i in pp:
        v = np.arange(i)
        k = np.concatenate((v,v))
        m = np.concatenate ((m,k))
    
        if v.shape !=0:
            s = np.array(n[-(v.shape[0]-1)::]) +1
            l = np.concatenate ( ( s , np.array([0]) ) )
            r = np.concatenate ((l,l+1))
            n = np.concatenate ((n,r))
        else:
            n = np.concatenate ((0,1))
    if group%2 != 0:        
        mn = np.array([m,n])
        mn = mn[:,0:-v.shape[0]]
    else:
        mn = np.array([m,n])
        
    return mn 

def LGFarFieldGouyPhase(mode_index):
    p = mode_index[0,:] # Controls Laguerre polynomial degree --> number of zeros -- radial
    l = mode_index[1,:] # Modify phase -- azimutal
     
    N = ( l + (2*p) ) * np.pi/2
    PSI = np.exp(-1j*N) #Gouy phase positive or negative ???
    
    return(PSI)

def applyphase(Ein, phase):
    #Ein --> M,N,N ; phase --> M
    return(Ein * phase[:,None,None])

@njit(parallel=True)
def LGmodes_CPU_parallel(w0,X,Y,mode_index,LG):
    #Compute just one part of the piramid --> complex conjugate must be done later
    RHO = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    LGpols = LG
    #Modes
    p = mode_index[0] # Controls Laguerre polynomial degree --> number of zeros
    l = mode_index[1] # Modify phase

    Emn = np.zeros( (mode_index.shape[1],X.shape[0],X.shape[0]) , np.complex64)
    
    for m in range(p.shape[0]):
        
        LG = LGpols[m]
        aa = (np.exp( (-l[m]*PHI)*1j) ) /w0
        bb = np.power(RHO/w0,l[m])
        cc = np.exp(-(RHO**2) / (w0**2))
    
        E = aa * bb * cc * LG
        Emn[m,...] = E / (np.sqrt(np.sum(np.abs(E)**2)))

    return Emn

#LG poly is a numpy array
def LGmodes_GPU(w0, X, Y, mode_index, LG, modeType = 'numpy'):
    
    """ 
        Computes LG modes on the GPU :
            w0 -- radius of the field (Mode field diammeter /2)
            X aand Y -- Cloud  of points that represent the space  (meshgrid in cartesian coordinates)
            mode_index -- 2D array with the modes indexes, first row = p, second raw = l
                p = polynomial degree
                l = modified phase
            LG -- pre-computed Laguerre polynomials -- can be an numpy array or cupy array
            LGtype -- specify the output array type
    
    """

    RHO,PHI = cart2pol(X,Y)
    
    p = mode_index[0,:,None,None]
    l = mode_index[1,:,None,None]
    RHO = RHO[None,:,:]
    PHI = PHI[None,:,:]
    
    w0_gpu = cp.asarray(w0)
    RHO_gpu = cp.asarray(RHO).astype(cp.complex64)
    PHI_gpu = cp.asarray(PHI).astype(cp.complex64)
    
    p_gpu =  cp.asarray(p) # Controls Laguerre polynomial degree --> number of zeros
    l_gpu =  cp.asarray(l) # Modify phase
    
    #Check what type of the input LG polynomials ---> Send them to GPU if there were not
    if type(LG) == np.ndarray:
        LG_gpu = cp.asarray(LG).astype(cp.complex64)
    else:
        LG_gpu = LG
        
    # Emn_gpu = cp.zeros( (p.shape[0],X.shape[0],X.shape[0]) , cp.complex64)
    # print(Emn_gpu.dtype)
    
    # Emn_gpu = ( (cp.exp( (-l_gpu*PHI_gpu)*1j) )/w0_gpu) * \
    #     cp.power(RHO_gpu/w0_gpu,l_gpu) * \
    #         cp.exp(-(RHO_gpu**2) / (cp.power(w0_gpu,2))) * \
    #             LG_gpu 
                
    Emn_gpu = ( (cp.exp( (-l_gpu*PHI_gpu)*1j) ) /w0_gpu) * cp.power(RHO_gpu/w0_gpu,l_gpu) * cp.exp(-(RHO_gpu**2) / (cp.power(w0_gpu,2))) * LG_gpu 
    Emn_gpu = Emn_gpu / (cp.sqrt(cp.sum(cp.absolute(Emn_gpu)**2,(1,2))))[:,None,None]
    
    
    if modeType == 'cupy':
        return(Emn_gpu)
    elif modeType == 'numpy':
        return(cp.asnumpy(Emn_gpu))
    
        
def LGmodes_CPU(w0,X,Y,mode_index,LG):
    #Compute just one part of the piramid --> complex conjugate must be done later
    
    RHO,PHI = cart2pol(X,Y)
    RHO = RHO[None,:,:]
    PHI = PHI[None,:,:]
               
    #Modes
    p = mode_index[0,:,None,None] # Controls Laguerre polynomial degree --> number of zeros
    l = mode_index[1,:,None,None] # Modify phase
    
    Emn = np.zeros( (mode_index.shape[1],X.shape[0],X.shape[0]) , np.complex64)
    
    # x = ((RHO**2) * (2 / w0**2)) #Argument of the genlaguerre function
    # LG = sp.eval_genlaguerre(p,l,x)
    
    LG = LG
    aa = ( (np.exp( (-l*PHI)*1j) ) /w0)
    bb = np.power(RHO/w0,l)
    cc = np.exp(-(RHO**2) / (w0**2))
    
    Emn[:mode_index.shape[1],:,:] = ne.evaluate("aa * bb * cc * LG")     
    
    Emn = Emn / (np.sqrt(np.sum(abs(Emn)**2,(1,2))))[:,None,None]
    
    return(Emn)


@njit(parallel=True)
def ComputeAllLGmodes_list_parallel ( LGmodes, indexes):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.

    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    - returns a list with all the modes
    """
    WholeModesSet = []
    LGmodesConjugate = np.conjugate(LGmodes)
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    for count,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet.append(LGmodes[count,...])
        else:
            #Pair of modes
            WholeModesSet.append(LGmodes[count,...])
            WholeModesSet.append(LGmodesConjugate[count,...])

    #Done for Loop  
    return(WholeModesSet)

@njit(parallel=True)
def ComputeAllLGmodes_array_parallel ( LGmodes, indexes  ):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.

    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    - return an arry with all the modes
    """

    LGmodesConjugate = np.conjugate(LGmodes) #conjugate all the input modes
    
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    l = len(n)
    uniqueM = len(np.where(n==0)[0])
    num_modes = (l - uniqueM)*2 + uniqueM
    
    WholeModesSet = np.zeros((num_modes,LGmodes.shape[1],LGmodes.shape[2]),np.complex64)
    count = 0;
    for idx,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet[count,...] = (LGmodes[idx,...])
            count += 1
        else:
            #Pair of modes
            WholeModesSet[count,...] = (LGmodes[idx,...])
            count += 1
            WholeModesSet[count,...] = (LGmodesConjugate[idx,...])
            count += 1

    #Done for Loop  
    return(WholeModesSet)

def ComputeAllLGmodes_array ( LGmodes, indexes):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.

    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    - return an arry with all the modes
    """

    LGmodesConjugate = np.conjugate(LGmodes) #conjugate all the input modes
    
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    l = len(n)
    uniqueM = len(np.where(n==0)[0])
    num_modes = (l - uniqueM)*2 + uniqueM
    
    WholeModesSet = np.zeros((num_modes,LGmodes.shape[1],LGmodes.shape[2]),np.complex64)
    count = 0;
    for idx,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet[count,...] = (LGmodes[idx,...])
            count += 1
        else:
            #Pair of modes
            WholeModesSet[count,...] = (LGmodes[idx,...])
            count += 1
            WholeModesSet[count,...] = (LGmodesConjugate[idx,...])
            count += 1

    #Done for Loop  
    return(WholeModesSet)

def ComputeAllLGmodes_list ( LGmodes, indexes):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.
    
    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    """
    WholeModesSet = []
    LGmodesConjugate = conjugate(LGmodes)
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    for count,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet.append(LGmodes[count,...])
        else:
            #Pair of modes
            WholeModesSet.append(LGmodes[count,...])
            WholeModesSet.append(LGmodesConjugate[count,...])
            
    #Done for Loop        
    return((WholeModesSet))

@njit(parallel=True)
def ComputeAllLGmodesFarField_array_parallel ( LGmodes, indexes, Gouy ):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.

    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    - return an arry with all the modes
    """

    LGmodesConjugate = np.conjugate(LGmodes) #conjugate all the input modes
    
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    l = len(n)
    uniqueM = len(np.where(n==0)[0])
    num_modes = (l - uniqueM)*2 + uniqueM
    
    WholeModesSet = np.zeros((num_modes,LGmodes.shape[1],LGmodes.shape[2]),np.complex64)
    count = 0;
    for idx,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet[count,...] = (LGmodes[idx,...]) * Gouy[idx]
            count += 1
        else:
            #Pair of modes
            WholeModesSet[count,...] = (LGmodes[idx,...]) * Gouy[idx]
            count += 1
            WholeModesSet[count,...] = (LGmodesConjugate[idx,...]) * Gouy[idx]
            count += 1

    #Done for Loop  
    return(WholeModesSet)

def ComputeAllLGmodesFarField_array ( LGmodes, indexes, Gouy):
    """
    Computes the comlex conjugate of the LGmodes if it is needed.

    - LGmodes matrix (Modes,X,Y).
    - Index of the half piramid, from LGindexes(modeGroup)function.
    - return an arry with all the modes
    """

    LGmodesConjugate = np.conjugate(LGmodes) #conjugate all the input modes
    
    #LGmn modes --> Unique modes n = 0
    n = indexes[1,:]
    l = len(n)
    uniqueM = len(np.where(n==0)[0])
    num_modes = (l - uniqueM)*2 + uniqueM
    
    WholeModesSet = np.zeros((num_modes,LGmodes.shape[1],LGmodes.shape[2]),np.complex64)
    count = 0;
    for idx,n_idx in enumerate(n):
        if (n_idx == 0):
            #Independent mode
            WholeModesSet[count,...] = (LGmodes[idx,...]) * Gouy[idx]
            count += 1
        else:
            #Pair of modes
            WholeModesSet[count,...] = (LGmodes[idx,...]) * Gouy[idx]
            count += 1
            WholeModesSet[count,...] = (LGmodesConjugate[idx,...]) * Gouy[idx]
            count += 1

    #Done for Loop  
    return(WholeModesSet)

######################## MODES GENERATION FUNCTIONS END #######################
##############  ##############  ##############  ##############  ############## 


##################### LAGUERRE POLYNOMIALK GENERATION #########################
##############  ##############  ##############  ##############  ############## 

#Function to be run with ipyparallel that give you the LG polynomials

def eval_genlaguerreCPU(p,l,x):
    
    return  sp.eval_genlaguerre(p,l,x)

def LGpol(p,l):
    o = scipy.special.eval_genlaguerre(p,l,x)
    return o


def eval_genlaguerreCPU_parallel(p,l,x):
    """ 
    NOTE : IPCLUSTER MUST BE LUNCH!!!
    Compute Laguerre polinomials using Ipyparallel:
        p -- is de degree of the Laguerre polynomials
        l -- is the coef that modulate the polynomial
        x -- the points where the polynomial wants to be evaluated
        returns an array in the GPU
    """
    try:
        rc = Client()
        dview = rc[:] # get all the 
    except TimeoutError:
        print('Try to laucnh the cluster : ipcluster start -n (num of cores)')
    
    #Make modules visible insides the cluster
    with dview.sync_imports():
         import numpy
         import scipy.special
         import scipy
        
    dview.push(dict(x = x))# make x parameter visible to all the cores
    LGpoly = dview.map_sync(LGpol,p,l)
    
    return(np.array(LGpoly)) #return a numpy array
    
    
    
def Okernel(p,l,k):
    with np.errstate(divide='ignore'):
        o = sp.factorial(p+l) / ( sp.factorial(k) * sp.factorial(p-k) * sp.factorial(l+k) )
    return(o)

def eval_genlaguerreGPU(p,l,x) :
    """ 
    Compute Laguerre polinomials in the GPU:
        p -- is de degree of the Laguerre polynomials
        l -- is the coef that modulate the polynomial
        x -- the points where the polynomial wants to be evaluated
        returns an array in the GPU
    """
    maxK = int(p.max())
    
    Xmatrix = cp.zeros( (maxK+1,x.shape[0],x.shape[1]), float )
    k = np.arange(0,maxK+1,1, int)
    k_gpu = cp.arange(0,maxK+1,1, int)
    x_gpu = cp.asarray(x).astype(cp.float) #Argument of the LG function
    Xmatrix = cp.power( x_gpu, k_gpu[:,None,None] )  * cp.power((-1),k_gpu[:,None,None])
    Xmatrix.shape
    
    N = int( p.shape[0])
    O = np.zeros((k.shape[0],N))
    O.shape
    
    for i in k:
        O[i,:] = Okernel(p,l,i)
    O[O == np.inf] = 0
    
    O_gpu = cp.asarray(O).astype(cp.float)
    
    #Mem checking
    meminfo = cp.cuda.Device(0).mem_info #Tuple (free,total)
    temp_nbytes = x.shape[0]**2 * (maxK+1) * N * 64 / 8 # dim x dim x DIM x DiM (4 dim array) - float64 beeing use / bytes
    LG_test_nbytes = x.shape[0]**2 * N * 64 / 8
    memfree = meminfo[0]
    memNeed = temp_nbytes + LG_test_nbytes
    
    print('Mem. avaliable ', memfree/1024**3, ' mem. needed ', memNeed/1024**3, ' in Gb')
    
    if(memfree > memNeed):
        temp = O_gpu[:,:,None,None] * Xmatrix[:,None,...]
        #print(temp.nbytes/1024**3)
        LG_test = cp.sum(temp,axis = 0)
        del temp
        cp._default_memory_pool.free_all_blocks()
    else:
        print('Performing it in blocks ...')
        #Do it in chuncks
        LG_test = cp.zeros((N,x.shape[0],x.shape[1]),float)
        # No smart : do it in # chuncks --> 
        chuncks = 4 # Hardcoded to 2 since I am not going to go that high in modegroups
        ch = N//chuncks
        rr = N%ch
        for i in range(chuncks):
            lowLim = i*ch
            
            if chuncks == i+1: # We are in the last iteration
                highLim = ch*(i+1) + rr
            else:
                highLim = ch*(i+1)
            
            temp = O_gpu[: , lowLim : highLim , None , None] * Xmatrix[:,None,...]
            LG_test[lowLim : highLim , ...] = cp.sum(temp, axis = 0)
            
            #print('Indexing from ', lowLim, ' to ', highLim , ' out ', N)
            
            del temp
            cp._default_memory_pool.free_all_blocks()
        print('Done')


    
    #Xmatrix is a 3D array, dimenssion depends on p and resolution of the modes.
    #O_gpu is 2D array with all possible coefs that multiply Xmatrix.
    
    #here I should implement how to do it in chucks in case temp gets so big
        # del temp
        #cp._default_memory_pool.free_all_blocks()
        #cp.cuda.Device(0).mem_info
        #floats64 cupy array are 8 bytes -> probably change to float 32
        #temSize = O_gpu.shape(1) * Xmatrix.shape * 8
    
    #temp = O_gpu[:,:,None,None] * Xmatrix[:,None,...] #This take a lot of memory, but I can be compute in chuncks+
    # print(temp.shape)
    # print(O_gpu.shape, Xmatrix.shape)
    # print(temp.nbytes)
    # print(temp.dtype)
   #LG_test = cp.sum(temp,axis = 0)

    
    #del temp
    #cp._default_memory_pool.free_all_blocks()
    
    return LG_test

################### LAGUERRE POLYNOMIALK GENERATION END #######################
##############  ##############  ##############  ##############  ##############


############### LG POLY + MODE GEN FUNCTIONS COMPACTATION #####################
##############  ##############  ##############  ##############  ##############

#This functions are more ready to use. Give you an extra level of abstraction

#COMPACT ALL ABOVE GENERATION FUNCTION IN ONE DEPENDING OF SOME FLAGS
def LGmodes(w0,X,Y,mode_index, engine = 'GPU', multicore = True):
    
    """
    Compute LGmodes of a given mode field diamter/2 and XY grid for some 
    mode_index coeficients. Computation can be done in the GPU or in
    the CPU{serial or parallel}. Check the inbuilt functions for more info
    
    """
    
    #Compute some commom variables : LG poly argument
    #coefs are needed
    p = mode_index[0]
    l = mode_index[1]
    #space is needed
    RHO,PHI = cart2pol(X,Y)
    lgArg = ((RHO**2) * (2 / w0**2))
    #NOTE : There are thing can be simplied I know... I am repeting things...
    # I am reusing some stuff
    
        
    if engine == 'GPU':
        print('Engine : GPU')
        LGpolynomials = eval_genlaguerreGPU(p,l,lgArg)
        mm = LGmodes_GPU(w0, X, Y, mode_index, LGpolynomials) 
        
    else: #then target 
    
        if multicore == True:
            print('Engine : CPU multicore')
            LGpolynomials = eval_genlaguerreCPU_parallel(p,l,lgArg)
            mm = LGmodes_CPU_parallel(w0,X,Y,mode_index,LGpolynomials)
        else:
            print('Engine : CPU singlecore')
            #Make arrays compatible for vectorization using single core
            pp = p[:,None,None]
            ll = l[:,None,None]
            llgArg = lgArg[None,...]
            LGpolynomials = eval_genlaguerreCPU(pp,ll,llgArg)
            mm = LGmodes_CPU(w0,X,Y,mode_index,LGpolynomials)
            
    return(mm)

def computeWholeSetofModes(modes_array,indexes, multicore = True):
    
    if multicore == True:
        return( ComputeAllLGmodes_array_parallel(modes_array,indexes) )
    else:
        return(ComputeAllLGmodes_array ( modes_array, indexes))
        
def computeWholeSetofModesFarField(modes_array,indexes, Gouy, multicore = True):
    
    if multicore == True:
        return( ComputeAllLGmodesFarField_array_parallel(modes_array, indexes, Gouy) )
    else:
        return(ComputeAllLGmodesFarField_array( modes_array, indexes, Gouy))


if __name__ == "__main__":
    
    t = times
    import mark_lib as mkl
    
    samples = 1024
    px_mmf = 0.3
    w0 = 15.3/2
    x = arange(-samples//2,samples//2,1) * px_mmf
    X,Y = meshgrid(x,x)
    
    #Modes: This just compute half of the piramid
    mode_group_max = 21
    index = graded_index_fiber_coefs(mode_group_max)
    
    # t.tic()
    # m,p = mkl.modes.LGnm_fiber(1565,w0,X,Y,index)
    # t.toc()
    t.tic()
    mm = LGmodes(w0,X,Y,index, engine = 'GPU', multicore = True)
    t.toc()
    # t.tic()
    # mmm,ppp = LGmodes(w0,X,Y,index, engine = 'CPU', multicore = True)
    # t.toc()
   

    
    
    
    
    
    
    
    
    
    #Test speed
    #mempool = cp.get_default_memory_pool()
    #pinned_mempool = cp.get_default_pinned_memory_pool()
    # t.tic()
    # LGmodesgpu = LGmodes(w0,X,Y,index,engine = 'GPU', multicore = True)
    # t.toc()
    
    # t.tic()
    # LGmodesCPUparallel = LGmodes(w0,X,Y,index,engine = 'CPU', parallel = True)
    # t.toc()
    
    # t.tic()
    # LGmodesCPUparallel = LGmodes(w0,X,Y,index,engine = 'CPU', parallel = False)
    # t.toc()
        
    # RHO,PHI = cart2pol(X,Y)
    # x = ((RHO**2) * (2 / w0**2))
    
    # #Pure GPU
    # LGpolynomialsGPU = eval_genlaguerreGPU(index[0],index[1],x)
    # LGmodes = LGmodes_GPU(w0,X,Y,index,LGpolynomialsGPU)
    
    # #LGpolynomialsCPU = eval_genlaguerreCPU_parallel(index[0],index[1],x)
    
    # LGWholeSet = ComputeAllLGmodes_array ( LGmodes, index)
    