import numpy as np 
import modelgenerator as mg 
import unscentedtransform as ut

def fallingBodyStatePropogator(state):
    
    g = -9.8
    dt = .1
    A = np.array([[1,dt],[0,1]])
    B = np.eye(2)
    u = np.array([0,g*dt]) # 2x1 col vec
    return np.matmul(A,state) + np.matmul(B , u)

