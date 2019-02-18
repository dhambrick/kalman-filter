import numpy as np
import unscentedtransform as ut

class ModelGenerator:
    def __init__(self,transitionFunction,kappa,alpha,beta , NoiseCov):
       self.uTransform = ut.UnscentedTransform(transitionFunction,kappa,alpha,beta)
       self.transitionFunction = transitionFunction
       self.NoiseCov = NoiseCov

    def calcNewState(self,oldState,oldStateCov):
        #TODO: Add logic to check dimensions of state variable
        self.newState = self.uTransform.estimateTransformedGaussian(oldState,oldStateCov)
        self.transformedSigmaPoints = self.uTransform.transformedSigmas
        self.newState["cov"] = self.newState["cov"] + self.NoiseCov
        return self.newState