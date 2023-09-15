
from pylab import *

def doOverlap(a,b):
    a = a / sqrt(sum(abs(a)**2))
    b = b / sqrt(sum(abs(b)**2))
    
    ov = average(a*conj(b))
    return ov 

def complexrand(a):
    m = zeros(a,complex64)
    m = abs(random(a)) * exp(1j*(random(a)*2*pi))
    return(m)

def complexrand_orto(a):
    M = complexrand(a)
    M_norm = sqrt(sum(abs(M)**2,1))
    M = M / M_norm[:,None]
    u, s, vh = np.linalg.svd(M, full_matrices=True)
    M = matmul(u,vh)
    M_norm = sqrt(sum(abs(M)**2,1))
    M = M / M_norm[:,None]
    
    U = matmul(M, transpose(conjugate(M)) )
    
    return(M,U)

def make_MMF_SI_MTM(ng, pols = 2):
    modesMax = sum(arange(ng)+1)
    if pols == 1 or pols == 2:
        MTM, U = complexrand_orto((modesMax*pols,modesMax*pols))
        return(MTM)
    else:
        print("pols must be 1 or 2")
    
def make_MMF_GI_MTM(ng,pols = 2):
    modesMax = sum(arange(ng)+1)
    if pols == 1 or pols == 2:
        MTM = zeros((modesMax*pols,modesMax*pols),complex64)
        m = arange(1,ng+1,1)*pols
        tm = 0
        for i,g in enumerate (m):  
            MTM[tm:tm+g,tm:tm+g], U = complexrand_orto((g,g))
            tm = m[i] + tm    
        #make sure that the overall phase between both pols is the same, a phase offset will imply that one pols is travelling a longer path
        #if pols == 2:
        #    MTM_locked = zeros_like(MTM)
        #    MTM_locked[:,::2] = MTM[:,::2] #store H as it is
        #    Aphi = angle(doOverlap(MTM[:,::2], MTM[:,1::2]))
        #    MTM_locked[:,1::2] = MTM[:,1::2] * exp(-1j*Aphi)
        #    MTM = MTM_locked
        #make MTM rows unitary power   : no need it because complexrand_orto is normalized already    
        #MTM_norm = sqrt(sum(abs(MTM)**2,1))
        #MTM = MTM / MTM_norm[:,None]       
        return(MTM)
    else:
        print("pols must be 1 or 2")


if __name__ == '__main__':

    print('hi')