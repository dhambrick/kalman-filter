import numpy as np 
import modelgenerator as modelgen

class UnscentedKalmanFilter:
    def __init__(self , procModel , sensorModel):
        self.procModel = procModel
        self.sensorModel = sensorModel

    def predict(self,oldState ,oldStateCov):
        self.newState = self.procModel.calcNewState(oldState,oldStateCov)
        self.newMeasurement = self.sensorModel.calcNewState(newState["mean"],newState["cov"])
        
        self.transformedStateSigmas = self.procModel.transformedSigmas
        self.transformedMeasurementSigmas = self.sensorModel.transformedSigmas
        self.covWeights = self.procModel.covWeights
        
        predictions = {"newState":self.newState ,"newMeasure":self.newMeasurement}
        return predictions
    
    def calcKGain(self,newState,predictedMeasure):
        N = np.max(newState["mean"].shape
        self.crossCovStateMeasure = np.zeros(N)
        for i in xrange(2*N+1):
            tempTerm1 = self.transformedStateSigmas[i] - newState["mean"]
            tempTerm2 = self.transformedMeasurementSigmas[i] - predictedMeasure["mean"]
            term = self.covWeights[i] * np.outer(tempTerm1 , tempTerm2)
            self.crossCovStateMeasure = self.crossCovStateMeasure + term
        self.K = np.matmul(self.crossCovStateMeasure ,np.linalg.inv(predictedMeasure["cov"]))    
        return self.K
    
    def correct(self,measModel, measurement):
        self.stateEstimate = self.stateEstimate + np.matmul(self.K , measurement - np.matmul(measModel.H,self.stateEstimate))
        self.P = self.P - np.matmul(self.K , np.matmul(measModel.H,self.P))
        print self.stateEstimate.shape
        return self.stateEstimate.reshape((1,2))
