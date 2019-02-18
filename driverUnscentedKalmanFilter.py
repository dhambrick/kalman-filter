import unscentedkalmanfilter as ukf
import unscentedtransform as ut 
import modelgenerator as modgen
import numpy as np
import matplotlib.pylab as plt
import fallingbodymodel as fbm 
import fallingsensormodel as fsm

#Set the Filter and model parameters
N = 100
kappa = 0
alpha = 1e-4
beta = 2
procCov = .25*np.eye(2)
sensorCov = .1*np.eye(2)

#create the process and sensor models
fallingBodyModel = modgen.ModelGenerator(fbm.fallingBodyStatePropogator , kappa , alpha , beta,procCov)
fallingBodySensorModel = modgen.ModelGenerator(fsm.fallingBodySensorMeasurement , kappa , alpha , beta,sensorCov)

#Create an unscented kalman filter instance
unscentedKalmanFilter = ukf.UnscentedKalmanFilter(fallingBodyModel ,fallingBodySensorModel)


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