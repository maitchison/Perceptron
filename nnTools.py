"""
    Tools to help neural nets.  Includes graphic, and column reduction
"""

import pylab as plt
import numpy as py
import math as math

class nnTools:
    # Show inputs on a graph
    def ShowInputs(self):
        
        plt.plot(self.inputs[:,0],self.inputs[:,1],'ro')
        
        plt.title('Input values')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.grid(True)        
        plt.show()

    # Show targets on a graph
    def ShowTargets(self):
        for i in range(0,self.nData):
            classifiction = self.targets[i]
            if (classifiction >= 0.5): 
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'ro')
            else:
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'go')                        

        plt.title('Targets')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.grid(True)        
        plt.show()

    # Estimates the gradient of the function at given location
    # todo: generalise 
    def EstimateGrad(self,x,y,epsilon = 0.01):        
        this = self.Predict(py.array([x,y]))
        dx = (this-self.Predict(py.array([x+epsilon,y])))/epsilon 
        dy = (this-self.Predict(py.array([x,y+epsilon])))/epsilon        
        return py.array([dx,dy])        

    # Show decision boundaries with targets on a graph
    def ShowBoundaries(self):
        
        if (self.nIn != 2):
            print("Boundary display only works with 2 features.")
            return
        
        density = 100
        for xlp in range(density):
            for ylp in range (density):
                x = xlp/(density-1)
                y = ylp/(density-1)
                grad = self.EstimateGrad(x, y)
                slope = py.dot(grad,grad)                
                if slope >= 0.1:               
                    plt.plot(x,y,'ko')                                                                        
        
        for i in range(0,4):
            classifiction = self.targets[i]
            if (classifiction == 1): 
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'ro')
            else:
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'go')
                
            plt.setp(v,'markersize',20.0)

        plt.title('Decision Boundaries')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.grid(True)        
        plt.show()
        
        # Removes columns one by one to see if removing them improves the performance
    # of the function or makes it worse.  This can be quite a slow process.
    # Todo: this could use a tidy up.  Would be better to delete columns rather than just 
    # zero them out.
    def AutoColumnDrop(self,trainingSet,trainingTarget,testingSet,testingTarget,message=""):
        
        self.Setup(trainingSet, trainingTarget)
        
        columns = self.nIn
        
        baseScore = self.TrialScore(trainingSet,trainingTarget,testingSet,testingTarget)
        
        print(message,"Beginning column drop data with ",columns," columns")    

        results = [False] * columns

        for i in range(columns):
            modifedTrainingSet = trainingSet
            modifedTrainingSet[:,i] = 0
            thisScore = self.TrialScore(modifedTrainingSet,trainingTarget,testingSet,testingTarget)
            if (thisScore > (baseScore + 1)):
                print(message,"Removing column ",i," resulted in a significant improvement.  Dropping this column and trying children...")
                self.AutoColumnDrop(modifedTrainingSet,trainingTarget,testingSet,testingTarget,message+"["+str(i)+"]")
                
        print(message,"<< finished")
        
    
    # Show's predictions 
    def ShowField(self):

        if (self.nIn != 2):
            print("Field display only works with 2 features.")
            return
        
        density = 25
        for xlp in range(density):
            for ylp in range (density):
                x = xlp/(density-1)
                y = ylp/(density-1)
                classifiction = self.Predict([x, y])
                if (classifiction >= 0.90): 
                    plt.plot(x,y,'rx')
                else:
                    plt.plot(x,y,'gx')                                                                        
        
        for i in range(self.nData):
            classifiction = self.targets[i]
            if (classifiction >= 0.5): 
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'ro')
            else:
                v = plt.plot(self.inputs[i,0],self.inputs[i,1],'go')
                
            plt.setp(v,'markersize',20.0)

        plt.title('Decision Boundaries')
        plt.xlabel('x')
        plt.ylabel('y')        
        plt.grid(True)        
        plt.show()

