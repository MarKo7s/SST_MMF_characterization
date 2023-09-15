from numpy import linalg as la
from numpy import conj
from numba import jit, prange
import numpy as np

# nearest Symetric Positive Semi-Definite

parallel = True
nogil = True
fastmath = True
cache = True


@jit(nopython = True, parallel = parallel, fastmath = fastmath, cache = cache)
def nearestSPSD_batch(A,tol = 0.01):
    
    num_matrices = A.shape[0]
    #A = A.astype(np.complex64)
    near_A = np.zeros(A.shape, np.complex64)
    
    tol = tol
    for i in prange(num_matrices):
        near_A[i,:,:] = nearestSPSD(A[i,:,:], tol = tol)

    return(near_A)

#It does not check for symetry
@jit(nopython = True, parallel = parallel, fastmath = fastmath)
def nearestPSD_batch(A,tol = 0.01):
    
    num_matrices = A.shape[0]
    #A = A.astype(np.complex64)
    near_A = np.zeros(A.shape, np.complex64)
    
    tol = tol
    for i in prange(num_matrices):
        near_A[i,:,:] = nearestPSD(A[i,:,:], tol = tol)

    return(near_A)


@jit(nopython = True, nogil = nogil, fastmath = fastmath, cache = cache)
def nearestSPSD(A, tol = 0.01):
    """Find the nearest symetric positive semi-definite real or complex matrix input

    John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    #A = A.astype(np.complex64)
    
    B = (A + conj(A.T)) / 2
    _, s, V = la.svd(B)

    S = np.diag(s).astype(np.complex64)
    _H = S.dot(V)
    H = np.dot(conj(V.T), _H)

    A2 = (B + H) / 2

    A3 = (A2 + conj(A2.T)) / 2

    check_PSD = isPSD(A3, tol)
    if check_PSD == True:
        return A3
    else:
        
        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not isPSD(A3,tol):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

@jit(nopython = True, nogil = nogil, fastmath = fastmath, cache = cache)
def nearestPSD(A, tol = 0.01):
    """Find the nearest positive semi-definite matrix from a already
        symetric matrix

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    

    check_PSD = isPSD(A, tol)
    if check_PSD == True:
        return A
    else:
        
        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not isPSD(A,tol):
            mineig = np.min(np.real(la.eigvals(A)))
            A += I * (-mineig * k**2 + spacing)
            k += 1

        return A


@jit(nopython = True, nogil = nogil, fastmath = fastmath, cache = cache)
def isPSD(B, tol):
    """Check is it is positve semi-definite via eigendecomposition"""
    eVals = (la.eigvals(B))
    eVals_real = (eVals.real)
    #print(eVals_real.dtype)
    PSD_check = np.all(eVals_real > -tol)
    #print(PSD_check)
    if PSD_check == True : 
        return True
    else:
        return False
    

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
    
if __name__ == '__main__':
    
    from matplotlib.pyplot import imshow, show
    import os
    import sys
    sys.path.append("C:\LAB\Coding\Python\MODULES")
    import mark_lib as mkl
    t = mkl.times
    
    test_matrix = np.random.rand(1,100,100).astype(np.complex64)
    t.tic()
    A = nearestSPSD_3D(test_matrix)
    #A = nearestSPSD(test_matrix)
    #r = isPSD(test_matrix, 0.01)
    t.toc()
    test_matrix = np.random.rand(1000, 500,500).astype(np.complex64)
    t.tic()
    #A = nearestSPSD(test_matrix)
    A = nearestSPSD_3D(test_matrix)
    #r = isPSD(test_matrix, 0.01)
    t.toc()
    
    #imshow(A)    
    #show()