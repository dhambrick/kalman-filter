import numpy as np 
import modelgenerator as mg 
import unscentedtransform as ut

def fallingBodySensorMeasurement(state):
    H = .75*np.eye(2)
    return np.matmul(H,state) 


