import unscentedkalmanfilter as ukf
import unscentedtransform as ut 
import modelgenerator as modgen
import numpy as np
import matplotlib.pylab as plt
import fallingbodymodel as fbm 
import fallingsensormodel as fsm

#Set the Filter and model parameters
N = 100
dt = .1
kappa = 0
alpha = 1e-4
beta = 2
procCov = np.eye(2)
sensorCov = .1*np.eye(2)
t = np.zeros(N) # 1D vector
x0v0 = np.array([100,0]) #initial state estimate
P0  = np.zeros((2,2)) #initial state covariance


#create the process and sensor models
fallingBodyModel = modgen.ModelGenerator(fbm.fallingBodyStatePropogator , kappa , alpha , beta,procCov)
fallingBodySensorModel = modgen.ModelGenerator(fsm.fallingBodySensorMeasurement , kappa , alpha , beta,sensorCov)

#Create an unscented kalman filter instance
unscentedKalmanFilter = ukf.UnscentedKalmanFilter(fallingBodyModel ,fallingBodySensorModel)

stateEstimates = np.zeros((N,2))
measurements = np.zeros((N,2))
P = []
P.append(procCov)
stateEstimates[0] = x0v0
trueState = x0v0
x = []
o =  []
x.append(0)
o.append(0)
#Start the time loop
for i in range(1,N):
    t[i] = t[i-1] + dt
    #Create a new observation
    observationNoiseMean = np.zeros(2)
    observationNoiseCov = sensorCov
    observationNoiseSample = np.random.multivariate_normal(observationNoiseMean ,observationNoiseCov )
    newObservation = fsm.fallingBodySensorMeasurement(trueState) + observationNoiseSample
    trueState = fbm.fallingBodyStatePropogator(trueState)
    
    #Execute the kalman filter 
    # 1: predict  
    predictions = unscentedKalmanFilter.predict(stateEstimates[i-1],P[i-1])
    
    # 2: Calculate K gain
    K = unscentedKalmanFilter.calcKGain(predictions["newState"],predictions["newMeasure"])
    # 3: Correct Prediction with K gain and sensor observation
    correctedEstimate = unscentedKalmanFilter.correct(newObservation) 
    
    stateEstimates[i] = correctedEstimate["state"]
    P.append(correctedEstimate["cov"])

print stateEstimates
plt.figure()
plt.plot(t,stateEstimates[0])
plt.show()