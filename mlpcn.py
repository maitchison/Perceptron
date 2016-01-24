"""
    A Multi layered PCN Neural Net
"""

import pylab as plt
from numpy import * 
import math as math
from nnBase import *

class LayeredPcn(nnBase):

    def Setup(self,inputs,outputs):
        
        super(LayeredPcn, self).Setup(inputs,outputs)
        
        self.thresholdFunction = ThresholdFunction.Sigmoid            
        
        # Initialize network
        self.weights = py.random.rand(self.nIn+1,self.nOut)*0.1-0.05
        #self.weights = py.random.rand(self.nIn+1,self.nOut)*0.1-0.05
                
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
    def Train(self, learningSpeed, maxIterations = 1000, bailError = 0.0001):        
                
        if py.shape(self.targets)[0] != py.shape(self.inputs)[0]:
            print ("Shape mismatch, found ", py.shape(self.inputs)," input shape and ", py.shape(self.targets),  " target shape.")
            return False
        
        if self.logging:
            print("Starting training on data set:")
            print("Number of samples = ",self.nData)        
            print("Number of features = ",self.nIn)
            
        weightDecay = 1;
        momentium = 0.5;         
        
        delta = zeros(shape(self.weights))   
        
        for n in range(maxIterations):                      
              
            self.outputs = self.Forward(self.inputs);
                        
            dEdO = self.targets-self.outputs
            dOdN = self.outputs * ( 1.0 - self.outputs)
            dNdW = self.inputs
                                
            dEdW = transpose(dot(transpose(dEdO * dOdN), dNdW))  
                        
            #print("E",dEdO[0],"O",dOdN[0],"N",dNdW[0])            
            #print("outputs",self.outputs)
                                    
            delta = (momentium * delta) + (math.pow((1-momentium),3) * learningSpeed * dEdW)
            
            self.weights += delta 

            
            self.weights = self.weights * weightDecay                             
            
            """
            deltao = (self.targets-self.outputs) * self.outputs * (1.0-self.outputs)
            #deltah = self.inputs * (1.0-self.inputs)*(py.dot(deltao,transpose(self.weights)))
            
            updatew2 = zeros((shape(self.weights)))
            #updatew1 = learningSpeed*(dot(transpose(self.inputs),deltah[:,:-1]))
            updatew2 = learningSpeed*(dot(transpose(self.inputs),deltao))
                        
            self.weights -= updatew2
            
            #self.weights -= learningSpeed*py.dot(py.transpose(self.inputs),self.targets-self.outputs)
                              
            """                     
            error = self.StdError()            
                                    
            if self.logging and (n % 10 == 0):
                print("Iteration:",n," error = ",error)
                
            if (error < bailError):
                break
            
        if self.logging:
            print("Performed ",n," iterations. With a final error of ",self.StdError())
            
        #py.set_printoptions(formatter={'float': lambda x: format(x, '2.2')})
        #print(self.outputs)                
            
        return True

