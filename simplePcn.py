import pylab as plt
import numpy as py
import math as math
from nnBase import * 

# Base class for neural nets
class simplePcn(nnBase):
    
    
    def Setup(self,inputs,outputs):
                    
        super(simplePcn, self).Setup(inputs,outputs)
        
        # Initialize network
        self.weights = py.random.rand(self.nIn+1,self.nOut)*0.1-0.05
                
        # Add bias values
        self.inputs = self.ConcatBias(self.inputs)
        
    # Push inputs through network. 
    def Forward(self,inputs):
        outputs = py.dot(inputs,self.weights)
        return self.Threshold(outputs)        
    
    # Trains the network on input data
    # learningSpeed: the speed at which to learn.  Some small value [0.1..0.3] works in most cases.
    # maxIterations: the maximum number of iterations to perform.
    # bailError: When error is below this amount the training will stop
    def Train(self, learningSpeed, maxIterations = 1000, bailError = 0.5):        
                
        if py.shape(self.targets)[0] != py.shape(self.inputs)[0]:
            print ("Shape mismatch, found ", py.shape(self.inputs)," input shape and ", py.shape(self.targets),  " target shape.")
            return False
        
        if self.logging:
            print("Starting training on data set:")
            print("Number of samples = ",self.nData)        
            print("Number of features = ",self.nIn)
        
        for n in range(maxIterations):    
                                
            self.outputs = self.Forward(self.inputs);
            self.weights += learningSpeed*py.dot(py.transpose(self.inputs),self.targets-self.outputs)
                                    
            error = self.StdError()
                                    
            if self.logging and (n % 10 == 0):
                print("Iteration:",n," error = ",error)
                
            if (error < bailError):
                break
            
        if self.logging:
            print("Performed ",n," iterations. With a final error of ",self.StdError())                
            
        return True


