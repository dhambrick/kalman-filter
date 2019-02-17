import kalman 
import numpy as np
import matplotlib.pylab as plt


g = -9.8
N = 100
dt = .1

xvState = np.zeros((N,2)) # N rows by 2 cols
t = np.zeros(N) # 1D vector
x0v0 = np.array([100,0])
A = np.array([[1,dt],[0,1]])
u =  np.transpose(np.array([[0,g*dt]])) # 2x1 col vec
xvState[0] = x0v0
modelFreeFall = kalman.ProcessModel(A,np.eye(2),u,xvState , 0.8)
oldState = modelFreeFall.state[0].reshape((2,1))
z = np.zeros((N,2))
H = np.eye(2)
measurementNoise = 0.9
sensor = kalman.MeasurementModel(2*H,measurementNoise,z)
kalman = kalman.KalmanFilter(np.zeros((2,2)))
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
#plt.scatter( t, np.transpose(sensor.z)[0])
plt.scatter(t,np.transpose(modelFreeFall.state)[0])
plt.scatter(t[1:100],np.transpose(stateEstimates)[0])
#plt.scatter(t[1:100],detP)

plt.show()