import pylab as plt
import numpy as py
import math as math

class ThresholdFunction:
    Square, Sigmoid = range(2)

class pcn:
    
    # Creates a Perceptron neural network. 
    # inputs: an n sized array containing the input vectors.
    # targets: an n sized array containing the true answers for given input.    
    def __init__(self,inputs,targets):    
        self.logging = False
        self.Setup(inputs,targets)
        
            
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
    
        # Initialize network
        self.weights = py.random.rand(self.nIn+1,self.nOut)*0.1-0.05
        
        # Store data
        self.inputs = inputs
        self.targets = targets
        
        # Add bias values
        self.inputs = self.ConcatBias(self.inputs)
        
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
        
        correctAnswers = 0
        for i in range(samples):
            if (results[i] == testTargets[i]):
                correctAnswers = correctAnswers + 1

        if self.logging:
            print("Correctly guessed ",correctAnswers, " out of ",samples,"=",correctAnswers/samples*100,"%")
        
        return correctAnswers/samples
        
    # Prints out the current weights
    def ShowWeights(self):
        print("Weights are \n",self.weights)
        
        
    # Runs the system over test data multiple times giving a final score of how
    # well the system performed.
    def TrialScore(self,trainingData,trainingTargets,testData,testTargets):
                        
        trialIterations = 1000
        score = py.empty([trialIterations])
        
        for i in range(trialIterations):
            self.Setup(trainingData,trainingTargets)
            self.Train(0.25,100,0.2)
            score[i] = self.Test(testData, testTargets)*100                
                            
        mean = py.mean(score)
        std = py.std(score)                            
        print("Completed ",trialIterations, " tests.  mean correctness = ",mean,"% Standard Deviation:",std)
        
        return mean
        

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

    
    # Push inputs through network. 
    def Forward(self,inputs):
        outputs = py.dot(inputs,self.weights)
        return self.Threshold(outputs)        
    
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


    # Apparently this "add's the inputs that match the bias node."
    """
    def confmat(self,inputs,targets):

        
        inputs = py.concatenate((inputs,-py.ones((self.nData,1))),axis=1)
        
        outputs = py.dot(inputs,self.weights)
    
        nClasses = py.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = py.where(outputs>0,1,0)
        else:
            # 1-of-N encoding
            outputs = py.argmax(outputs,1)
            targets = py.argmax(targets,1)

        cm = py.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = sum(py.where(outputs==i,1,0)*py.where(targets==j,1,0))

        print (cm)
        print (py.trace(cm)/sum(cm))
        
        

    # I'm not really sure what this function does??        
    def logic(self):


        a = py.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
        b = py.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

        p = pcn(a[:,0:2],a[:,2:])
        p.Train(a[:,0:2],a[:,2:],0.25,10)
        p.confmat(a[:,0:2],a[:,2:])

        q = pcn(a[:,0:2],b[:,2:])
        q.Train(a[:,0:2],b[:,2:],0.25,10)
        q.confmat(a[:,0:2],b[:,2:])
        """
