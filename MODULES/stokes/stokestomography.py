import sys
import pathlib
p = pathlib.Path(__file__).parent.parent
path_to_module = p
sys.path.append(str(path_to_module))

from pylab import *
from einsumt import einsumt
from numba import jit
import numexpr as ne

from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from stokes.PSD import nearestSPSD_batch, nearestPSD_batch


try:
    import cupy as cp
    cupy_available = True
except ModeleNotFoundError:
    cupy_available = False

def GellManMatrices(N):
    lamb = zeros((N**2 - 1,N,N), complex64) # -1 because we are removing identity 
    k_vec = arange(N-1) + 2 # Vector containing the entries for the ones we calculate the norm
    #This matrices are diagonal,  1)lam[0] has 2 elements!=0, 2)lam[1] has 3 elements !=0 and so on
    norm = sqrt(2/(k_vec*(k_vec-1))) #Values for (k-1)-th entry --> Scanining k=1 to k=N-1
    norm_ = norm * -1*(k_vec-1) #Values of k-th entry --> This is just on the last element
    #Built the matrices:
    for k in range(N-1):
        fill_diagonal(lamb[k,:k+1,:k+1],norm[k]) #fill diagonal of the matrix(k) from 0 to k-1 staring with k=1
        lamb[k,k+1,k+1] = norm_[k] #Fill single elemente k of the diagonal of the matrix(k) 
        #print(lamb[k])
    idx = k + 1 #borrow the indexing
   #Built the rest of the matrices:
    for j in range(N):
       for k in range(N):
            if k<j:
                lamb[idx,j,k]= 1
                lamb[idx,k,j]= 1
                idx+=1
                lamb[idx,j,k]= 1j
                lamb[idx,k,j]= -1j
                idx+=1
    print("Num. of Matrices",idx)
    return(lamb)

def GellManMatrices_sparse(N):
    lamb = lil_matrix( (N**2 - 1, N**2), dtype = complex64 ) # Each GellManMatrix has to be flattern before indexing
    lamb_tmp = zeros((N,N), dtype = complex64) # temporal storage of the target matrix is being constructed
    k_vec = arange(N-1) + 2 # Vector containing the entries for the ones we calculate the norm
    #This matrices are diagonal,  1)lam[0] has 2 elements!=0, 2)lam[1] has 3 elements !=0 and so on
    norm = sqrt(2/(k_vec*(k_vec-1))) #Values for (k-1)-th entry --> Scanining k=1 to k=N-1
    norm_ = norm * -1*(k_vec-1) #Values of k-th entry --> This is just on the last element
    #Built the matrices:
    for k in range(N-1):
        fill_diagonal(lamb_tmp[:k+1,:k+1],norm[k]) #fill diagonal of the matrix(k) from 0 to k-1 staring with k=1
        lamb_tmp[k+1,k+1] = norm_[k] #Fill single elemente k of the diagonal of the matrix(k) 
        lamb[k,:] = lamb_tmp.ravel()
        lamb_tmp.fill(0)
    idx = k + 1 #borrow the indexing
   #Built the rest of the matrices:
    for j in range(N):
       for k in range(N):
            if k<j:
                Nidx1 = (N*j)+k
                Nidx2 = (N*k)+j
                
                lamb[idx,Nidx1] = 1
                lamb[idx,Nidx2] = 1
                idx+=1
                lamb[idx,Nidx1] = 1j
                lamb[idx,Nidx2] = -1j
                idx+=1
                
    lamb = lamb.tocsr() # change to csr formation           
    print("Num. of Matrices",idx)
    return(lamb)


#This one is proven to work
def StokesTomography_BeanchMark(N):
    lamb = GellManMatrices(N)
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count,N),complex64) # Each stoke state is a vector of dimension N
    StokesWeight = zeros((GellMann_count,StokesStates_count),float32) #For each GellMann Matrices we have a vector if weight of the size of StokesStates to contrsuct the generalised Stokes Vector
    print("StokesStates:", StokesStates.shape)
    print("StokesWeight:", StokesWeight.shape)
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    W,V = linalg.eig(lamb) #Gellman Matrices are Hermitian so I could use eigh though the order is inverted, I can be bother to order them
    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    for GellMann_idx in range(GellMann_count):
        #W,V = linalg.eig(lamb[GellMann_idx]) #W[idx] eigenvalues ; V  eigenvectors[:,idx]
        for eigIdx, eigValue in enumerate(W[GellMann_idx]):
            if eigValue != 0:
                TargetState =  V[GellMann_idx,:,eigIdx];
                v = abs(sum(StokesStates * conj(TargetState),1))**2; #This is expensive for large StokesStates
                redundantIdx = where(v>(1 - 1.0/N**2))[0]

                if redundantIdx.shape[0] == 0:
                    StokesStates[stokes_idx] = TargetState #New fresh eigenVector
                    redundantIdx = stokes_idx
                    stokes_idx+=1 # Update
                    
                StokesWeight[GellMann_idx, redundantIdx] =  StokesWeight[GellMann_idx, redundantIdx] + (eigValue.real) #if the eigenVector was already add up the weight
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
                
    return(lamb,StokesStates,StokesWeight)

#Main function --> Only working changes are introduced
def StokesTomography_BenchMark_v2(N):
    lamb = GellManMatrices(N)
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count,N),complex64) # Each stoke state is a vector of dimension N
    StokesWeight = zeros((GellMann_count,StokesStates_count),float32) #For each GellMann Matrices we have a vector if weight of the size of StokesStates to contrsuct the generalised Stokes Vector
    print("StokesStates:", StokesStates.shape)
    print("StokesWeight:", StokesWeight.shape)
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    W,V = linalg.eig(lamb) #Gellman Matrices are Hermitian so I could use eigh though the order is inverted, I can be bother to order them
    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    for GellMann_idx in range(GellMann_count):
        #W,V = linalg.eig(lamb[GellMann_idx]) #W[idx] eigenvalues ; V  eigenvectors[:,idx]
        for eigIdx, eigValue in enumerate(W[GellMann_idx]):
            if eigValue != 0:
                TargetState =  V[GellMann_idx,:,eigIdx];
                v = abs(sum(StokesStates[0:stokes_idx] * conj(TargetState),1))**2; #Overlap with the stakes already filled
                redundantIdx = where(v>(1 - 1.0/N**2))[0]
                #print('Gell:', GellMann_idx,'Index:', eigIdx, 'ReundantIdx:', redundantIdx, 'Value:', v)
                if redundantIdx.shape[0] == 0:
                    StokesStates[stokes_idx] = TargetState #New fresh eigenVector
                    redundantIdx = stokes_idx
                    stokes_idx+=1 # Update
                    
                StokesWeight[GellMann_idx, redundantIdx] =  StokesWeight[GellMann_idx, redundantIdx] + (eigValue.real) #if the eigenVector was already add up the weight
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
                
    return(lamb,StokesStates,StokesWeight)

def StokesTomography(N):
    lamb = GellManMatrices(N)
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count,N),complex64) # Each stoke state is a vector of dimension N
    StokesWeight = zeros((GellMann_count,StokesStates_count),float32) #For each GellMann Matrices we have a vector if weight of the size of StokesStates to contrsuct the generalised Stokes Vector
    print("StokesStates:", StokesStates.shape)
    print("StokesWeight:", StokesWeight.shape)
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    print('Extracting eigenvalues and eigenVectors ...')
    W,V = linalg.eig(lamb) #W[idx] eigenvalues (these are real) ; V  eigenvectors[:,idx]
    print('Done')
    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    print('Getting the unique eigenvectors ...')
    for GellMann_idx in range(GellMann_count):
        WW = W[GellMann_idx]
        W_non_zero_idx = where(WW != 0)[0]
        W_non_zero_values = WW[W_non_zero_idx]
        potential_new_states = len(W_non_zero_values)
        TargetStates =  V[GellMann_idx,:,W_non_zero_idx]
        if stokes_idx > 0:
            v = abs(matmul((conj(TargetStates)),transpose(StokesStates[0:stokes_idx])))**2; #Overlap with the already filled stokesStakes
            redundantIdxs = where(v>(1 - 1.0/N**2)) #Index of the redundant states
            #For each evaluated TargetState--> There are to options: 1) original 2) redundant
            redundantIdY = redundantIdxs[0] # Indexing into the W_non_zero_values that have being redundant detected
            redundantIdX = redundantIdxs[1] # Indexing into the StokesWeights
            newIdxs = arange(0, v.shape[0]) # All possible indexing states
            newIdxs = delete(newIdxs, redundantIdY) #Delete the entry of the redundant states
            redundant_states_count = len(redundantIdY) #How many are reduntant
            new_states_count = potential_new_states - redundant_states_count  #How many are new
            #Let's take care of the redundant States:
            StokesWeight[GellMann_idx, redundantIdX] += (W_non_zero_values[redundantIdY].real) #if the eigenVector was already add up the weight 
        else:
            new_states_count = potential_new_states #At the first iteration all are good
            newIdxs = arange(len(W_non_zero_idx))
            #print(stokes_idx, (stokes_idx+new_states_count),newIdxs , StokesStates.shape, TargetStates.shape)
        #The rest are new StokesStates (new ones):
        if new_states_count != 0:
            StokesStates[stokes_idx:(stokes_idx+new_states_count)] = TargetStates[newIdxs,:] #New fresh eigenVector
            StokesWeight[GellMann_idx, stokes_idx:(stokes_idx+new_states_count)] += (W_non_zero_values[newIdxs].real) #if the eigenVector was already add up the weight
            stokes_idx+=new_states_count # Update
            
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
                
    return(lamb,StokesStates,StokesWeight)

#Main version for sparse matrices: Changes are introduced after checking that non-sparse version works
def StokesTomography_sparse(N):
    lamb = GellManMatrices_sparse(N) #GellmanMatrices in sparse csr format ( matrix are flattern as a 1D array)
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count, N), dtype = complex64) # I am thinking to compress the StokesStates as well, let see how it performs
    StokesWeight_shape = (GellMann_count,StokesStates_count)
    StokesWeight = lil_matrix( StokesWeight_shape, dtype = float32 ) # Using lil_matrix as sparse constructor for simplicity
    print("StokesStates:", StokesStates.shape)
    print("StokesWeight:", StokesWeight_shape)

    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    print('Getting the unique eigenvectors ...')
    for GellMann_idx in range(GellMann_count):    
        W,V = linalg.eig(lamb[GellMann_idx,:].toarray().reshape((N,N))) #IN: Needs 2D GellmanMatrix NO SPARSE - OUT:W[idx] eigenvalues ; V  eigenvectors[:,idx]
        W_non_zero_idx = where(W != 0)[0] #Get only the non 0 eignValues
        W_non_zero_values = W[W_non_zero_idx] #Get their index
        potential_new_states = len(W_non_zero_values)
        TargetStates =  transpose(V[:,W_non_zero_idx]) 
        if stokes_idx > 0:
            v = abs(matmul((conj(TargetStates)),transpose(StokesStates[0:stokes_idx])))**2; #Overlap with the already filled stokesStakes
            redundantIdxs = where(v>(1 - 1.0/N**2)) #Index of the redundant states
            #For each evaluated TargetState--> There are to options: 1) original 2) redundant
            redundantIdY = redundantIdxs[0] # Indexing into the W_non_zero_values that have being redundant detected
            redundantIdX = redundantIdxs[1] # Indexing into the StokesWeights
            newIdxs = arange(0, v.shape[0]) # All possible indexing states
            newIdxs = delete(newIdxs, redundantIdY) #Delete the entry of the redundant states
            redundant_states_count = len(redundantIdY) #How many are reduntant
            new_states_count = potential_new_states - redundant_states_count  #How many are new
            #Let's take care of the redundant States:
            StokesWeight[GellMann_idx, redundantIdX] += (W_non_zero_values[redundantIdY].real) #if the eigenVector was already add up the weight 
        else:
            new_states_count = potential_new_states #At the first iteration all are good
            newIdxs = arange(len(W_non_zero_idx))
        #The rest are new StokesStates (new ones):
        if new_states_count != 0:
            #print(stokes_idx, (stokes_idx+new_states_count),newIdxs , StokesStates.shape, TargetStates.shape)
            StokesStates[stokes_idx:(stokes_idx+new_states_count)] = TargetStates[newIdxs, :] #New fresh eigenVector
            StokesWeight[GellMann_idx, stokes_idx:(stokes_idx+new_states_count)] += (W_non_zero_values[newIdxs].real) #if the eigenVector was already add up the weight
            stokes_idx+=new_states_count # Update
            
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
        
    StokesWeight = StokesWeight.tocsr()            
    return(lamb,StokesStates,StokesWeight)


# Testing version (non-sparse)
def StokesTomography_testing(N):
    lamb = GellManMatrices(N)
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count,N),complex64) # Each stoke state is a vector of dimension N
    StokesWeight = zeros((GellMann_count,StokesStates_count),float32) #For each GellMann Matrices we have a vector if weight of the size of StokesStates to contrsuct the generalised Stokes Vector
    print("StokesStates:", StokesStates.shape)
    print("StokesWeight:", StokesWeight.shape)
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    print('Extracting eigenvalues and eigenVectors ...')
    W,V = linalg.eig(lamb) #W[idx] eigenvalues (these are real) ; V  eigenvectors[:,idx]
    print('Done')
    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    print('Getting the unique eigenvectors ...')
    for GellMann_idx in range(GellMann_count):
        WW = W[GellMann_idx]
        W_non_zero_idx = where(WW != 0)[0]
        W_non_zero_values = WW[W_non_zero_idx]
        potential_new_states = len(W_non_zero_values)
        TargetStates =  V[GellMann_idx,:,W_non_zero_idx]
        if stokes_idx > 0:
            v = abs(matmul((conj(TargetStates)),transpose(StokesStates[0:stokes_idx])))**2; #Overlap with the already filled stokesStakes
            redundantIdxs = where(v>(1 - 1.0/N**2)) #Index of the redundant states
            #For each evaluated TargetState--> There are to options: 1) original 2) redundant
            redundantIdY = redundantIdxs[0] # Indexing into the W_non_zero_values that have being redundant detected
            redundantIdX = redundantIdxs[1] # Indexing into the StokesWeights
            newIdxs = arange(0, v.shape[0]) # All possible indexing states
            newIdxs = delete(newIdxs, redundantIdY) #Delete the entry of the redundant states
            redundant_states_count = len(redundantIdY) #How many are reduntant
            new_states_count = potential_new_states - redundant_states_count  #How many are new
            #Let's take care of the redundant States:
            StokesWeight[GellMann_idx, redundantIdX] += (W_non_zero_values[redundantIdY].real) #if the eigenVector was already add up the weight 
        else:
            new_states_count = potential_new_states #At the first iteration all are good
            newIdxs = arange(len(W_non_zero_idx))
            #print(stokes_idx, (stokes_idx+new_states_count),newIdxs , StokesStates.shape, TargetStates.shape)
        #The rest are new StokesStates (new ones):
        if new_states_count != 0:
            StokesStates[stokes_idx:(stokes_idx+new_states_count)] = TargetStates[newIdxs,:] #New fresh eigenVector
            StokesWeight[GellMann_idx, stokes_idx:(stokes_idx+new_states_count)] += (W_non_zero_values[newIdxs].real) #if the eigenVector was already add up the weight
            stokes_idx+=new_states_count # Update
            
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
                
    return(lamb,StokesStates,StokesWeight)


########### BROKEN DREAMS ################
#It work as slow as the pure python one
def StokesTomography_compiled(N):
    lamb = GellManMatrices(N)
    W,V = linalg.eig(lamb) #W[idx] eigenvalues ; V  eigenvectors[:,idx]
    StokesStates, StokesWeight, stokes_idx = calculateStokesStates((W.real).astype(float32),V, N)
    
    if stokes_idx == StokesStates.shape[0]:
        print("Stokes Projections found = ", stokes_idx, " -- SUCCESS")
    else:
        print("Somenthing went wrong finding the stokestates")
        print("Target States dimension = ", StokesStates.shape[0], "but ",stokes_idx, "found -- FAIL" )
                
    return(lamb,StokesStates,StokesWeight)

@jit(nopython = False, cache = False)
def calculateStokesStates(W,V, N):
    StokesStates_count = 2 * N**2 - N
    GellMann_count = N**2 - 1
    #GellMann matrices eigenValues are related with the StokesWeights and eigenVectors with the StokesStates
    StokesStates = zeros((StokesStates_count,N), dtype = complex64) # Each stoke state is a vector of dimension N
    StokesWeight = zeros((GellMann_count,StokesStates_count), dtype = float32) #For each GellMann Matrices we have a vector if weight of the size of StokesStates to contrsuct the generalised Stokes Vector
    #print("StokesStates:", StokesStates.shape)
    #print("StokesWeight:", StokesWeight.shape)
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    #Extract eigenvalues and eigenvectors of the GellMann Matrices
    stokes_idx = 0 #use to update the entry of the given StokesWeight and StokesState
    for GellMann_idx in range(GellMann_count):
        #W,V = linalg.eig(lamb[GellMann_idx]) #W[idx] eigenvalues ; V  eigenvectors[:,idx]
        for eigIdx in range(W[GellMann_idx].shape[0]):
            if W[GellMann_idx,eigIdx] != 0:
                TargetState =  V[GellMann_idx,:,eigIdx];
                v = np.abs(np.sum(StokesStates * conj(TargetState),1))**2;
                redundantIdx = np.where(v>(1 - 1.0/N**2))[0]
                          
                if redundantIdx.shape[0] == 0:
                    StokesStates[stokes_idx] = TargetState #New fresh eigenVector
                    redundantIdx = stokes_idx
                    stokes_idx+=1
                print(W[GellMann_idx,eigIdx].dtype)
                StokesWeight[GellMann_idx, redundantIdx] =  StokesWeight[GellMann_idx, redundantIdx] + (W[GellMann_idx,eigIdx]) #if the eigenVector was already add up the weight
    StokesStates = StokesStates/sqrt(2*N - 1)
    
    return(StokesStates, StokesWeight, stokes_idx)
    


def StokesVectorCalc(lamb, StokesWeights, StokeStatesMeasuredPower, ForcePSD = False, GPU = False, Sparse = True):
    
    """
    in:
            lamb (ndarray or csr-matrix): GellMann Matrices 
            StokesWeights (ndarray or csr-matrix): Weights in which the StokeStatesMeasuredPower has to be added
            StokeStatesMeasuredPower: Intensity measurement of each of the StokesStates projections as single vector(1D) or an array of vector(3D)
    
    out:
            Sn: Stokes Vector
            density Matrix
            eigenValues: of the density Matrix in descending order
            eigenVectors: of the density Matrix in descending order
            S0: Total intensity of the provided intensity measurements
    
    """
    
    I_dimension = len(StokeStatesMeasuredPower.shape)

    if I_dimension > 1:
        #numpy format is provided but we want to still take advantage of the sparcity
        if (Sparse == True or (isinstance(lamb, csr_matrix) and isinstance(StokesWeights, csr_matrix))):
            #If user is forcing sparse computation but numpy array was provided, cast to csr sparse
            if (isinstance(lamb, ndarray) or isinstance(StokesWeights, ndarray)):
                print('Casting arrays to csr matrix before calculation...')
                #asuming GellmanMatrices are provided as a 3D array:
                lamb = csr_matrix(reshape(lamb, (lamb.shape[0],-1))) #flattening the matrixes and compress
                StokesWeights = csr_matrix(StokesWeights)
            return StokesVector_vectorial_v3_sparse(lamb, StokesWeights,StokeStatesMeasuredPower, ForcePSD, GPU)
        else:
            return StokesVector_vectorial_v3(lamb,StokesWeights,StokeStatesMeasuredPower, ForcePSD, GPU)
    else:
        return StokesVector_single(lamb,StokesWeights,StokeStatesMeasuredPower)
    
    
def StokesVector_single(lamb,StokesWeights,StokeStatesMeasuredPower):
     
    #Normalization of the measurement so it is dependend of the bases number
    N = lamb.shape[1]
    S0 = sum(StokeStatesMeasuredPower)
    I_m = StokeStatesMeasuredPower/S0
    I_m = I_m * (2*N - 1)
    
    #Density Matrix
    k_nm = StokesWeights
    Sn = matmul(k_nm,I_m) #Stokes Vector
    densityMatrix = eye(N,N) / N
    densityMatrix = densityMatrix + sum(lamb * (0.5*Sn[:,None,None]),0) #Add everything including the I-matrix
    
    #EigenValues and EigenVectors:
    W,V = linalg.eig(densityMatrix) #W[idx] eigenvalues ; V  eigenvectors[:,idx]
    #Sort eigenVectors and eigenVector from higest to lowest
    sortIndexing = argsort(W) #lowest to largest
    sortIndexing = flip(sortIndexing) #largest to lowest
    eigenValues = W[sortIndexing]
    eigenVectors = V[:,sortIndexing]
    
    return(Sn, densityMatrix, eigenValues, eigenVectors, S0)


def StokesVector_vectorial(lamb,StokesWeights,StokeStatesMeasuredPower):
    
    N = lamb.shape[1]
    S0 = sum(StokeStatesMeasuredPower, 0)
    I_m = StokeStatesMeasuredPower / S0[None,...]
    I_m = I_m * (2*N - 1)

    #Density Matrix
    k_nm = StokesWeights
    Sn = einsumt('ijk ,li->ljk',I_m,k_nm) #multicore einsmt
    densityMatrix = eye(N,N) / N
    densityMatrix = densityMatrix[...,None,None] + einsumt('ijk ,ilm->jklm',lamb,0.5*Sn) #Add everything including the I-matrix

    #Extract eigvenValues and eigenVectors
    NN =  densityMatrix.shape[0]
    Nx = densityMatrix.shape[2]
    Ny = densityMatrix.shape[3]

    eigenValues = zeros((NN,Nx,Ny),complex64)
    eigenVectors = zeros((NN,NN,Nx,Ny),complex64)
    
    # I hate to have to loop...
    for i in range(Ny):
        for j in range(Nx):
            W,V = linalg.eig(densityMatrix[...,i,j]) #This can not be vectorize above 3 dimensions -- looping is not to slow -- sorting it may be worse
            sortIndexing = argsort(W) #lowest to largest
            sortIndexing = flip(sortIndexing) #largest to lowest
            eigenValues[...,i,j] = W[sortIndexing]
            eigenVectors[...,i,j] = V[:,sortIndexing]
    
    return(Sn, densityMatrix, eigenValues, eigenVectors, S0)

#Here the dimension of the StokeStatesMeasuredPower is reduce allowing to use matmul instead insumt to calculate the stokes vector -  x30 speed up for 128x128 pixel and 9 modegroups
#I need to try to reduce the Gellman matrices from 3D to 2D so I can use matmul as well
#EigenDecomposition is already at the max
def StokesVector_vectorial_v2(lamb,StokesWeights,StokeStatesMeasuredPower):
    
    N = lamb.shape[1]
    S0 = sum(StokeStatesMeasuredPower, 0)
    I_m = StokeStatesMeasuredPower / S0[None,...]
    I_m = I_m * (2*N - 1)
    I_m = transpose(reshape(I_m,(I_m.shape[0],-1)))

    #Compose the stokes Vector Sn
    k_nm = StokesWeights
    Sn = matmul(I_m,transpose(k_nm))

    #Reconstruct the density matrix
    I_Matrix = eye(N,N) / N
    densityMatrix = einsumt('ijk ,li->ljk',lamb,0.5*Sn) #Add everything 
    densityMatrix = I_Matrix[None,:,:] + densityMatrix #Include the I-matrix

    #EigenValues and EigenVector in order
    W,V = linalg.eig(densityMatrix)
    sortIndexing = argsort(W,axis=1) #lowest to largest
    sortIndexing = flip(sortIndexing,axis=1) #largest to lowest
    eigenValues = take_along_axis(W,sortIndexing,1)
    eigenVectors = take_along_axis(V,sortIndexing[:,None,:],axis = 2)
        
    return(Sn, densityMatrix, eigenValues, eigenVectors, S0)

#I pushed the Sn and densityMatrix to the limit, eigenValues still kind of slow --- Maybe splitting work in chuncks and launch threads it could speed up the sigendecompostion
def StokesVector_vectorial_v3(lamb, StokesWeights, StokeStatesMeasuredPower, ForcePSD = False, GPU = False):
    
    N = lamb.shape[1]
    S0 = sum(StokeStatesMeasuredPower, 0)
    #I_m = divide(StokeStatesMeasuredPower, S0[None,...], where = S0[None,...]!=0 )
    I_m = StokeStatesMeasuredPower / S0[None,...]
    I_m = I_m * (2*N - 1)
    I_m = transpose(reshape(I_m,(I_m.shape[0],-1)))#flattern pixel array - Y dimension correspond different pixels (Independed) - X correspond to the projection Intensity value

    #Compose the stokes Vector Sn
    print("StokesVector... ")
    k_nm = StokesWeights
    Sn = matmul(I_m, transpose(k_nm)) #I_m and K_nm if I am not wrong are always real values. Keep an eye an do not waist resources
    print("Done")

    #Reconstruct the density matrix
    print("DenstyMatrix... ")
    I_Matrix = eye(N,N) / N # this is for later
    lamb_dim = lamb.shape
    lamb2D = reshape(lamb, (lamb_dim[0],-1) ) #flattening the matrixes 
    densityMatrix = matmul(0.5*Sn,lamb2D) #Add everything 
    densityMatrix = reshape( densityMatrix, (densityMatrix.shape[0],lamb_dim[1],lamb_dim[2]) )
    densityMatrix = (ne.evaluate('I_Matrix + densityMatrix)')).astype(complex64)
    #densityMatrix = I_Matrix[None,:,:] + densityMatrix #Include the I-matrix
    print("Done")
    
    #!##################### Positive Semi-Definite ########################################
    #! Force the densityMatrices to be positive definite --> This would be the slowest part
    if ForcePSD == True:
        print('Forcing SPSD matrices')
        densityMatrix = nearestSPSD_batch(densityMatrix)
        print('Done')
    #!######################################################################################    
    #!###################################################################################### 
    
    print("Calculating eigenValues and eigenVectors...")
    #Check if GPU avaliable and flag is ON
    if GPU == True and cupy_available == True:
        print('GPU on')
        eigenValues,eigenVectors = sorted_eig_cp_memsafe(densityMatrix)
    #EigenValues and EigenVector in order
    else:
        eigenValues,eigenVectors = sorted_eig_np(densityMatrix)
        
    print("Done")
    S0 = S0.ravel()
    
    return(Sn, densityMatrix, eigenValues, eigenVectors, S0)

def StokesVector_vectorial_v3_sparse(lamb_s, StokesWeights_s, StokeStatesMeasuredPower, ForcePSD = False, GPU = False):
    
    N = int(sqrt(lamb_s.shape[1]))
    Num_lamb_s = lamb_s.shape[0]
    S0 = sum(StokeStatesMeasuredPower, 0)
    #I_m = divide(StokeStatesMeasuredPower, S0[None,...], where = S0[None,...]!=0 ) 
    I_m = StokeStatesMeasuredPower / S0[None,...]
    I_m = I_m * (2*N - 1)
    I_m = transpose(reshape(I_m,(I_m.shape[0],-1)))#flatten pixel array - Y dimension correspond different pixels (Independent) - X correspond to the projection Intensity value

    #Compose the stokes Vector Sn
    print("StokesVector... ")
    k_nm = StokesWeights_s
    Sn = csr_matrix.dot(I_m, k_nm.transpose()) #Equivalent to matmul --> Sn = matmul(I_m, transpose(k_nm))
    print("Done")

    #Reconstruct the density matrix
    print("DenstyMatrix... ")
    I_Matrix = eye(N,N) / N # this is for later
    densityMatrix = (csr_matrix.dot(0.5*Sn, lamb_s)) #Matmul again --> return a density matrix per pixel
    densityMatrix = reshape( densityMatrix, (densityMatrix.shape[0], N ,N) ) #Reshape to compute eigs
    densityMatrix = (ne.evaluate('I_Matrix + densityMatrix')).astype(complex64)
    print("Done")
    
    #!##################### Positive Semi-Definite ########################################
    #! Force the densityMatrices to be positive definite --> This would be the slowest part
    if ForcePSD == True:
        print('Forcing SPSD matrices')
        densityMatrix = nearestSPSD_batch(densityMatrix)
        print("Done")
    #!######################################################################################    
    #!######################################################################################    

    print("Calculating eigenValues and eigenVectors...")
    #Check if GPU avaliable and flag is ON
    if GPU == True and cupy_available == True:
        print('GPU on')
        eigenValues,eigenVectors = sorted_eig_cp_memsafe(densityMatrix)
    #EigenValues and EigenVector in order
    else:
        eigenValues,eigenVectors = sorted_eig_np(densityMatrix)
        
    print("Done")
    S0 = S0.ravel()
    
    return(Sn, densityMatrix, eigenValues, eigenVectors, S0)


def sorted_eig_np(M):
    W,V = linalg.eig(M)
    sortIndexing = argsort(W,axis=1) #lowest to largest
    sortIndexing = flip(sortIndexing,axis=1) #largest to lowest
    eigenValues = take_along_axis(W,sortIndexing,1)
    eigenVectors = take_along_axis(V,sortIndexing[:,None,:],axis = 2)
    
    return(eigenValues, eigenVectors)

def sorted_eig_cp(M):
    #I need to check the memory first to see if it fits... if it does not fit you will have to loop
    densityMatrix_gpu = cp.asarray(M).astype(cp.complex64)
    W,V = cp.linalg.eigh(densityMatrix_gpu)
    del densityMatrix_gpu
    cp._default_memory_pool.free_all_blocks() #free mem on the GPU is not being used
    sortIndexing = cp.argsort(W,axis=1) #lowest to largest
    sortIndexing = cp.flip(sortIndexing,axis=1) #largest to lowest
    eigenValues = cp.take_along_axis(W,sortIndexing,1)
    eigenVectors = cp.take_along_axis(V,sortIndexing[:,None,:],axis = 2)
    
    eigenValues = cp.asnumpy(eigenValues)
    eigenVectors = cp.asnumpy(eigenVectors)
    
    #make sure to clean everything
    del W, V, sortIndexing
    cp._default_memory_pool.free_all_blocks() #free mem on the GPU is not being used
        
    return(eigenValues, eigenVectors)

def sorted_eig_cp_memsafe(Matrices):
    
    #stimate memory needed in the GPU:
    Mem_eigV = Matrices.shape[0] * Matrices.shape[1] * 8 
    Mem_eigVec = Matrices.shape[0] * Matrices.shape[1]**2 * 8
    Mem_needed_temp = Mem_eigV + Mem_eigVec
    MemNeeded = Matrices.nbytes + (Mem_needed_temp * 2) # factor of 2 for the eigh 
    cp._default_memory_pool.free_all_blocks() #free mem on the GPU is not being used
    GPUmemInfo = cp.cuda.Device(0).mem_info #Tuple (free,total)
    GPUMemAvaliable = GPUmemInfo[0]
    
    print(f'Memory needed: { MemNeeded / 1024**3} Gb, GPU mememory avaliable: {GPUMemAvaliable / 1024**3}')
    if MemNeeded < GPUMemAvaliable:
        
        return sorted_eig_cp(Matrices)
        
    else:
        
        print('Performing in blocks...')
        numBlocks = int32(ceil( MemNeeded / GPUMemAvaliable )) #
        
        Matrices_count = Matrices.shape[0]
        Matrices_dimension = Matrices.shape[1]
        
        blockSize = Matrices_count // numBlocks
        blocksize_r = Matrices_count % numBlocks #the remaining add them at the last iteration
        
        #Allocate mem. on RAM and perform in chuncks
        eigenValues = zeros((Matrices_count, Matrices_dimension),dtype = float64)
        eigenVectors = zeros((Matrices_count,Matrices_dimension,Matrices_dimension), dtype = complex64)
        
        for blockIdx in range(numBlocks):
            
            lowLim = blockIdx*blockSize
            if numBlocks == blockIdx+1:
                highLim = blockSize*(blockIdx+1) + blocksize_r
            else:
                highLim = blockSize*(blockIdx+1)
            
            eVal,eVec = sorted_eig_cp(Matrices[lowLim:highLim, ...])
            
            eigenValues[lowLim:highLim] = eVal
            eigenVectors[lowLim:highLim, ...] = eVec
        
        return eigenValues, eigenVectors
        
if __name__ == '__main__':

    #lamb, stokestates, stokesweights = StokesTomography(78)
    #projnum = stokestates.shape[0]
    #print(projnum)
    #px = 256
    #analyzerStateOutPOW = rand(projnum, px,px )
    #print(analyzerStateOutPOW.shape)
    #Sn, densityMatrix, eigenValues, eigenVectors, S0 = StokesVectorCalc(lamb,stokesweights,analyzerStateOutPOW, GPU = True)
    StokesTomography_compiled(5)