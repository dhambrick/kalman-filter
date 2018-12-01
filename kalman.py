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

g = -9.8
N = 100
dt = .1

xvState = np.zeros((N,2)) # N rows by 2 cols
t = np.zeros(N) # 1D vector
x0v0 = np.array([100,0])
A = np.array([[1,dt],[0,1]])
u =  np.transpose(np.array([[0,g*dt]])) # 2x1 col vec
xvState[0] = x0v0
modelFreeFall = ProcessModel(A,np.eye(2),u,xvState , 0.2)
oldState = modelFreeFall.state[0].reshape((2,1))
z = np.zeros((N,2))
H = np.eye(2)
measurementNoise = 0.9
sensor = MeasurementModel(2*H,measurementNoise,z)
kalman = KalmanFilter(np.zeros((2,2)))
stateEstimates = []
detP = []

for i in range(1,N):
    oldState = modelFreeFall.state[i-1].reshape((2,1))
    modelFreeFall.state[i] = np.transpose(modelFreeFall.calcNewState(oldState))
    trueState = modelFreeFall.state[i].reshape((2,1))
    measure = sensor.calcNewMeasurement(trueState)
    
    kalman.predict(oldState,modelFreeFall,sensor)
    kalman.calcKGain(sensor)
    stateEstimates.append(kalman.correct(sensor,measure))
    detP.append(np.linalg.det(kalman.P))


    z[i] = measure.reshape(2)
    t[i] = t[i-1] + dt
detP = np.array(detP)
stateEstimates =  np.array(stateEstimates).reshape((99,2))
print detP.shape,t.shape ,np.transpose(stateEstimates)[0].shape
plt.figure()
plt.scatter( t, np.transpose(sensor.z)[0])
plt.scatter(t,np.transpose(modelFreeFall.state)[0])
plt.scatter(t[1:100],np.transpose(stateEstimates)[0])
plt.scatter(t[1:100],detP)

plt.show()