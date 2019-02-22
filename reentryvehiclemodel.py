import numpy as np 
from scipy.integrate import odeint
import modelgenerator as mg 
import unscentedtransform as ut
import matplotlib.pyplot as plt

par = { "B0": -.59783 ,"H0":13.406 ,"Gm0": 3.9860e5 ,"R0":6374 ,"dt":.01 ,"Q":np.diag(np.array([2.4064e-5,2.4064e-5,0]))}

def G(state):
    r = np.sqrt(state[0]*state[0] + state[1]*state[1])
    return -par["Gm0"]/(r*r*r)

def B(x5):
    return par["B0"] * np.exp(x5)
def D(state):
    Rk = np.sqrt(state[0]*state[0]+state[1]*state[1])
    Vk = np.sqrt(state[2]*state[2]+state[3]*state[3])
    Bk = B(state[4])
    argExp = (par["R0"]- Rk) / par["H0"]
    return -Bk*np.exp(argExp) * Vk

def dynamicsODE(state , t ):
    noise = [0,0,0]
    dSdt = [state[2] , 
                state[3] , 
                D(state) * state[2] +G(state)*state[0] + noise[0] ,
                D(state) * state[3] +G(state)*state[1] + noise[1],
                noise[2]]
    return dSdt 

def discretizedDynamics(state , noise):
    newState = [state[0] + par["dt"]*state[2] , 
                state[1] + par["dt"] *state[3] , 
                state[2] + par["dt"] * (D(state) * state[2] +G(state)*state[0] + noise[0]) ,
                state[3] + par["dt"] * (D(state) * state[3] +G(state)*state[1] + noise[1]),
                state[4] + par["dt"] * noise[2]]
    return newState            

def reentryVehicleStatePropogater(state):
    stateNoiseMean = np.zeros(5)
    stateNoiseSample = np.random.multivariate_normal(stateNoiseMean ,par["Q"] )
    newState = discretizedDynamics(state,stateNoiseSample)
    return newState

