import pylab as plt
import math as math
from numpy import * 
from nnBase import *

# Two layered Neural Net
class TwoLayerNN(nnBase):

    def Setup(self,hiddenNodes,inputs,outputs,testInputs,testTargets):
        
        super(TwoLayerNN, self).Setup(inputs,outputs)
        
        self.thresholdFunction = ThresholdFunction.Sigmoid            
        
        self.nHidden = hiddenNodes
        
        # Initialize network weights
        # note: 1/sqrt is textbook but I seem to do much better with really small weights
        initialRange = 0.01/sqrt(self.nIn)
        self.weights1 = py.random.rand(self.nIn+1,self.nHidden)*initialRange*2-initialRange
        self.weights2 = py.random.rand(self.nHidden+1,self.nOut)*initialRange*2-initialRange
        
        self.testInputs = testInputs
        self.testTargets = testTargets
                
        # Add bias values
        self.inputs = self.ConcatBias(self.inputs)
        
        self.momentum = 0.9
 
    # Push inputs through network. 
    def Forward(self,inputs):
        
        # push inputs through first layer...
        self.h1 = self.Threshold(dot(inputs,self.weights1))
                    
        self.h1 = self.ConcatBias(self.h1)                
                    
        # push inputs through second layer...
        outputs = dot(self.h1,self.weights2)  
        
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
            
        updatew1 = zeros((shape(self.weights1)))
        updatew2 = zeros((shape(self.weights2)))                
        
        lastError = 0 
        step = 1
        jitter = 0
        errorTable = zeros(100) 
        
        lastTestResult = 0
        testCounter = 0
        bestTestScore = 0
        
        change = list(range(self.nData))
                    
        for n in range(maxIterations):                      
              
            self.outputs = self.Forward(self.inputs);
                    
            # find error derivative at output layer            
            deltaO = (self.outputs-self.targets) * self.outputs * ( 1.0 - self.outputs)                        
            deltaH = self.h1 * (1.0 - self.h1) * dot(deltaO,transpose(self.weights2))             
            
            updatew2 = step * -0.02*(dot(transpose(self.h1),deltaO)) + self.momentum*updatew2
            updatew1 = step * -0.02*(dot(transpose(self.inputs),deltaH[:,:-1])) + self.momentum*updatew1 
                        
            self.weights1 += updatew1
            self.weights2 += updatew2                                                                                                                                    
                                 
            error = self.StdError()
            errorTable[n % 100] = lastError - error
            
            jitter = std(errorTable)*100
            
            step = step * 0.99999                    
            
            if (n % 100 == 0):
                testResult = self.Test(self.testInputs, self.testTargets)
                bestTestScore = max(bestTestScore,testResult)
                
                if (testResult < lastTestResult):
                    testCounter += 1
                if (testResult > lastTestResult):
                    testCounter -= 1
                    if (testCounter < 0): 
                        testCounter = 0
                if (testCounter >= 6):                
                    print("Overtraining detected, stopping...")
                    return True   
                lastTestResult = testResult
                
                                                                
            if self.logging and (n % 1000 == 0):
                print("Iteration:",n," error = "+str(error)+ " step "+str(step)+ " variance "+str(jitter)+ " last test result="+str(testResult)+" best = "+str(bestTestScore)+" ["+str(testCounter)+"]")
            
            
            # Randomise order of inputs
            random.shuffle(change)
            self.inputs = self.inputs[change,:]
            self.targets = self.targets[change,:]  
            
                
            if (error < bailError):
                break
            
            lastError = error
            
        if self.logging:
            print("Performed ",n," iterations. With a final error of ",lastError)
            
        #py.set_printoptions(formatter={'float': lambda x: format(x, '2.2')})
        #print(self.outputs)                
            
        return True



