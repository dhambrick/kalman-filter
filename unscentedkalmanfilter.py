import numpy as np 
import modelgenerator as modelgen

class UnscentedKalmanFilter:
    def __init__(self , procModel , sensorModel):
        self.procModel = procModel
        self.sensorModel = sensorModel

    def predict(self,oldState ,oldStateCov):
        self.predictedState = self.procModel.calcNewState(oldState,oldStateCov)
        self.predictedObservation = self.sensorModel.calcNewState(self.predictedState["mean"],self.predictedState["cov"])
        self.transformedStateSigmas = self.procModel.transformedSigmaPoints
        self.transformedMeasurementSigmas = self.sensorModel.transformedSigmaPoints
        self.covWeights = self.procModel.covWeights
        self.meanWeights = self.procModel.meanWeights
        
        predictions = {"newState":self.predictedState ,"newMeasure":self.predictedObservation}
        return predictions
    
    def calcKGain(self,predictedState,predictedObservation):
        N = np.max(predictedState["mean"].shape)
        self.crossCovStateMeasure = np.zeros((N,N))
       
        for i in xrange(0,2*N+1):
       
            tempTerm1 = self.transformedStateSigmas[i] - predictedState["mean"]
            tempTerm2 = self.transformedMeasurementSigmas[i] - predictedObservation["mean"]
            term = self.covWeights[i] * np.outer(tempTerm1 , tempTerm2)
            self.crossCovStateMeasure = self.crossCovStateMeasure + term
        self.K = np.matmul(self.crossCovStateMeasure ,np.linalg.inv(predictedObservation["cov"]))    
        return self.K
    
    def correct(self, newObservation):
        correctedStateEstimate = self.predictedState["mean"] + np.matmul(self.K,(newObservation - self.predictedObservation["mean"]))
        correctedStateCov = self.predictedState["cov"] - np.matmul(np.matmul(self.K,self.predictedObservation["cov"]),np.transpose(self.K))
        update = {"state":correctedStateEstimate , "cov":correctedStateCov}
        return update               
    
    def run(self,stateEstimate,covEstimate, newObservation):
        if (newObservation is not None) and (stateEstimate is not None) and (covEstimate is not None):
            #Execute the kalman filter 
            # 1: predict  
            predictions = self.predict(stateEstimate,covEstimate)
            # 2: Calculate K gain
            K = self.calcKGain(predictions["newState"],predictions["newMeasure"])
            # 3: Correct Prediction with K gain and sensor observation
            correctedEstimate = self.correct(newObservation)
            return correctedEstimate 
        elif (newObservation is  None):
            print "No new obeservation: Only executing predict phase"
            #Execute the kalman filter 
            # 1: predict  
            predictions = self.predict(stateEstimate,covEstimate)
            statePrediction = {"state":predictions["newState"]["mean"] , "cov":predictions["newState"]["cov"]}
            return statePrediction
        
