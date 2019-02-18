import numpy as np 
import modelgenerator as modelgen

class UnscentedKalmanFilter:
    def __init__(self , procModel , sensorModel):
        self.procModel = procModel
        self.sensorModel = sensorModel

    def predict(self,oldState ,oldStateCov):
        self.newState = self.procModel.calcNewState(oldState,oldStateCov)
        self.newMeasurement = self.sensorModel.calcNewState(self.newState["mean"],self.newState["cov"])
        
        self.transformedStateSigmas = self.procModel.transformedSigmas
        self.transformedMeasurementSigmas = self.sensorModel.transformedSigmas
        self.covWeights = self.procModel.covWeights
        
        predictions = {"newState":self.newState ,"newMeasure":self.newMeasurement}
        return predictions
    
    def calcKGain(self,newState,predictedMeasure):
        N = np.max(newState["mean"].shape)
        self.crossCovStateMeasure = np.zeros((N,N))
        for i in xrange(0,2*N+1):
            tempTerm1 = self.transformedStateSigmas[i] - newState["mean"]
            tempTerm2 = self.transformedMeasurementSigmas[i] - predictedMeasure["mean"]
            term = self.covWeights[i] * np.outer(tempTerm1 , tempTerm2)
            self.crossCovStateMeasure = self.crossCovStateMeasure + term
        self.K = np.matmul(self.crossCovStateMeasure ,np.linalg.inv(predictedMeasure["cov"]))    
        return self.K
    
    def correct(self, newMeasurement):
        correctedStateEstimate = self.newState["mean"] + self.K*(newMeasurement["mean"] - self.newMeasurement)
        correctedStateCov = self.newState["cov"] - self.K*self.newMeasurement["cov"]*np.transpose(self.K)
        update = {"state":correctedStateEstimate , "cov":correctedStateCov}
        return update               
    
