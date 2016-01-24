"""
    Base class for neural networks
"""

import pylab as plt
import numpy as py
import math as math

class ThresholdFunction:
    Square, Sigmoid = range(2)

class nnBase:
    # Creates a Perceptron neural network. 
    # inputs: an n sized array containing the input vectors.
    # targets: an n sized array containing the true answers for given input.    
    def __init__(self):    
        self.logging = False    
            
    # Set's up input data for training
    # inputs: an n sized array containing the input vectors.
    # targets: an n sized array containing the true answers for given input.    
    def Setup(self,inputs,targets):            
        # Set up network size
        if py.ndim(inputs)>1:
            self.nIn = py.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if py.ndim(targets)>1:
            self.nOut = py.shape(targets)[1]
        else:
            self.nOut = 1
        
        self.nData = py.shape(inputs)[0]
        
        # Store data
        self.inputs = inputs
        self.targets = targets                    
        
        self.thresholdFunction = ThresholdFunction.Square
        

    # Calculate the error between the network's current state and targets.
    def Error(self):
        return py.dot(py.transpose(self.targets-self.outputs),self.targets-self.outputs)
    
    # Calculate square root of mean square error.
    def StdError(self):
        return math.sqrt((self.Error()/self.nData))

    # Returns a vector with a -1 column concatenated to the end.
    def ConcatBias(self,x):
        return py.concatenate((x,-py.ones((self.nData,1))),axis=1)

    # Runs a set of test data against the network and prints out the error.
    def Test(self, testInputs, testTargets):                
        
        samples = py.shape(testInputs)[0]
        
        if (py.shape(testTargets)[0] != samples):
            print("Sample length mismatch ",samples, py.shape(testTargets)[0])
            return
                    
        testInputs = self.ConcatBias(testInputs)
        
        results = self.Forward(testInputs)
        
        # stub: force 2 categories
        results[py.where(results[:]<0.5)] = 0
        results[py.where(results[:]>=0.5)] = 1
        
        correctAnswers = 0
        for i in range(samples):            
            if (results[i] == testTargets[i]):
                correctAnswers = correctAnswers + 1

        if self.logging:
            print("Correctly guessed ",correctAnswers, " out of ",samples,": error rate of ",100-(correctAnswers/samples*100),"%")
        
        return correctAnswers/samples
        
    # Prints out the current weights
    def PrintWeights(self):
        print("Weights are \n",self.weights)        
        
    # Runs the system over test data multiple times giving a final score of how
    # well the system performed.
    def TrialScore(self,trainingData,trainingTargets,testData,testTargets,trainingIterations = 100):
                        
        trialIterations = 100
        score = py.empty([trialIterations])
        
        for i in range(trialIterations):
            self.Setup(trainingData,trainingTargets)
            self.Train(0.15,trainingIterations,0.2)
            score[i] = self.Test(testData, testTargets)*100                
                            
        mean = py.mean(score)
        std = py.std(score)                            
        print("Completed ",trialIterations, " tests.  mean error = ",mean,"% Standard Deviation:",std)
        
        return mean
        

    def Train(self,inputs):
        raise NotImplementedError("Abstract method called.")
    
    def Forward(self,inputs):
        raise NotImplementedError("Abstract method called.")
    
    def sigmoid(self, x):
        return 1 / (1 + py.exp(-x))
    
    # Performs the threshold function inputs
    def Threshold(self, values):
        if (self.thresholdFunction == ThresholdFunction.Square):
            return py.where(values>0,1,0)
        elif (self.thresholdFunction == ThresholdFunction.Sigmoid):
            return self.sigmoid(values)
        return py.where(False,1,0)
            
    
    # Predict the value of a single input based on current weights    
    def Predict(self,input):
        input = self.ConcatBias(input)
        output = py.dot(input,self.weights)        
        return self.Threshold(output)[0]                                     