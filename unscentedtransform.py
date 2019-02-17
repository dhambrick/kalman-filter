import numpy as np 

class UnscentedTransform:
    def __init__(self,stateMean,stateCov,nonlinearTransform ,kappa,alpha,beta):
        self.stateDim = stateMean.shape
        self.stateMean = stateMean 
        self.stateCov = stateCov
        self.nlt = nonlinearTransform
        #UT Parameters
        self.N = np.max(self.stateDim)
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.Lambda = (self.alpha*self.alpha) * (self.N + self.kappa) - self.N  
        self.SigmaPoints = []
    
    def generateSigmaPoints(self):
        gamma = np.sqrt(self.N + self.kappa) 
        #In general we calculate  2N+1 sigma points where N is the dimension of the mean vector
        # The first Sigma Point is simply stateMean vector itself
        self.SigmaPoints.append(self.stateMean)
        S = gamma * np.linalg.cholesky(self.stateCov)
        #We can naively calculate the rest of the sigma points as follows:
        for i in xrange(0,self.N):
            self.SigmaPoints.append(self.stateMean + S[:,i])
            self.SigmaPoints.append(self.stateMean - S[:,i])
        self.SigmaPoints = np.array(self.SigmaPoints)
    def propagateSigmaPoints(self):
        vecNLT = np.vectorize(self.nlt)
        self.transformedSigmas = vecNLT(self.SigmaPoints)

    def generateWeights(self):
        #First we generate the weights for estimating the transformed mean
        meanW0 = self.Lambda/ (self.N + self.Lambda)
        meanW2N = .5* (1 / (self.N + self.Lambda))
        self.meanWeights = meanW2N * np.ones((2*self.N+1))
        self.meanWeights[0] = meanW0
        #Now we calculate the weights for estimating the transformed covariance
        #The only difference from the mean weights is that the initial weight covW0 as an extra term of 1-alpha^2 + beta added to it
        covW0 = meanW0 + (1-(self.alpha*self.alpha)+self.beta)
        covW2N = .5* (1 / (self.N + self.Lambda))
        self.covWeights = covW2N * np.ones((2*self.N+1))
        self.covWeights[0] = covW0 
    
    
    def estimateTransformedGaussian(self, stateMean , stateCov):
        #This method is where we put everything together and calculate the new state and covariance estimates
        #We give the option to recalculate the unscented transform by being able to override any previously defined 
        #state and covariance estimates
        #TODO: Add logic that either updates or self.N or enforces that the passed in state and covariance are of the same dimensions of the

        if stateMean is not None:
            self.stateMean = stateMean
        if stateCov is not None:
            self.stateCov = stateCov

        #Step 1: Calculate the sigma points
        self.generateSigmaPoints()
       
        #Step 2: Generate the weights
        self.generateWeights()
       
        #Step 3: Transform the Sigma points using the supplied Transform
        self.propagateSigmaPoints()
       
        #Step 4: Calculate the propogated mean state
        #Just for pedalogical purposes we implement this step using the naive mathematical formulation ie No numpy optimizations
        self.estimatedMean = np.zeros(self.N )
        for i in xrange(2*self.N+1):
            self.estimatedMean = self.estimatedMean + (self.transformedSigmas[i] * self.meanWeights[i])

        #Step 5: Calculate the propogated state covariance
        self.estimatedCovariance = np.zeros(self.stateCov.shape)
        for i in xrange(2*self.N+1):
            meanCenteredSigma = self.transformedSigmas[i] - self.estimatedMean
            self.estimatedCovariance = self.estimatedCovariance  + np.outer(meanCenteredSigma , meanCenteredSigma)
