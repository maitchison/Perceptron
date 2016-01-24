import os
import pylab as plt
from numpy import *
import time
from Perceptron import *
from TwoLayerNN import *
from ThreeLayerNN import *
from nnBase import ThresholdFunction

# Simple class to hold logic function inputs and targetse
class LogicFunctions:    
    INPUTS = array([[0,0],[0,1],[1,0],[1,1]])
    OR = array([[0],[1],[1],[1]])
    AND = array([[0],[0],[0],[1]])
    XOR = array([[0],[1],[1],[0]])
    
    
# Plots sample data where the first two columns are displayed x,y and the
# last column is taken as the category (0 or 1)
def Plot(input):
    # Plot the first two variables in our data.
    rows = shape(input)[0]
    cols = shape(input)[1]
    for i in range(rows):
        if input[i,cols-1] == 1:
            plt.plot(input[i,0],input[i,1],'rx')
        else:
            plt.plot(input[i,0],input[i,1],'go')
        
        plt.show()
    
# Load in testing data set.
pima = loadtxt('pima-indians-diabetes.data',delimiter=',')
print("Loaded PIMA data set, shape = ",shape(pima))

# Preprocess the data.

# This guy adds around +5%
pima[where(pima[:,0]>8),0] = 8

# This doesn't seem to help at all, maybe makes it a little worse?
# Actually it's very sensitive to when the boundaries.  My guess is that
# there is important information around for people in their 20's and how
# this gets separated makes a difference.
for i in range(1,10):
    min = (i)*10+11
    max = (i+1)*10+11
    pima[where((pima[:,7]<max) & (pima[:,7]>=min)),7] = i
        
# Normalize, again this doesn't seems to help at all with the mean accuracy
# but reduces the stdv of the error a lot.    
pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)

# drop column 
#pima[:,6] = 0

# Separate into two data sets
trainingSet = pima[::2,:8]
trainingTarget = pima[::2,8:9]

testingSet = pima[1::2,:8]
testingTarget = pima[1::2,8:9]
                  
#p = nnBase.nnBase(LogicFunctions.INPUTS,LogicFunctions.OR)
p = TwoLayerNN()
p.Setup(trainingSet,trainingTarget)
p.logging = True
p.Train(0.10, 100000)
p.Test(testingSet, testingTarget)
#p.TrialScore(trainingSet,trainingTarget,testingSet,testingTarget,1000)

# drop a random column
#p.AutoColumnDrop(trainingSet,trainingTarget,testingSet,testingTarget)

#p.thresholdFunction = ThresholdFunction.Square


#p.pcnfwd(inputs_bias,weights)