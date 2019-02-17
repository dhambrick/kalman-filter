import numpy as np 
import matplotlib.pylab as plt

class ProcessModel:
    def __init__(self,A,B,u , state , procStd):
        self.A = A
        self.B = B
        self.u = u
        self.state = state
        self.procStd = procStd
        self.procVar = self.procStd * self.procStd
        self.Q = self.procVar * np.eye(2) #covariance matrix of process model
    def calcNewState(self,oldState):
        #TODO: Add logic to check dimensions of state variable
        return np.matmul(self.A,oldState) + np.matmul(self.B , self.u) + self.procStd * np.random.randn(2,1) 

class MeasurementModel:
    def __init__(self,H,measureStd,z):
         self.H = H
         self.mStd = measureStd
         self.mVar = measureStd * measureStd
         self.R = self.mVar * np.eye(2)
         self.z = z
    def calcNewMeasurement(self,state):
        #TODO: Add logic to check dimensions of state variable
        return np.matmul(self.H , state) + self.mStd * np.random.randn(2,1) 

class KalmanFilter:
    def __init__(self , P0):
        self.P = P0
    def predict(self,oldState , procModel , measModel):
        self.stateEstimate = procModel.calcNewState(oldState)
        self.P = np.matmul( np.matmul(procModel.A , self.P),np.transpose(procModel.A))
    
    def calcKGain(self,measModel):
        H = measModel.H
        R = measModel.R 
        M = np.matmul(np.matmul(H , self.P),np.transpose(H)) + R
        invM = np.linalg.inv(M) 
        self.K = np.matmul(np.matmul(self.P ,np.transpose(H)),invM)
    
    def correct(self,measModel, measurement):
        self.stateEstimate = self.stateEstimate + np.matmul(self.K , measurement - np.matmul(measModel.H,self.stateEstimate))
        self.P = self.P - np.matmul(self.K , np.matmul(measModel.H,self.P))
        print self.stateEstimate.shape
        return self.stateEstimate.reshape((1,2))

