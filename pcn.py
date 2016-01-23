import numpy as py
#from statsmodels.sandbox.nonparametric.tests.ex_smoothers import weights

class pcn:
    
    def __init__(self,inputs,targets):
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
    
        # Initialise network
        self.weights = py.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self,inputs,targets,eta,nIterations):
        # Add the inputs that match the bias node
        inputs = py.concatenate((inputs,-py.ones((self.nData,1))),axis=1)
        # Training
        change = range(self.nData)        

        for n in range(nIterations):
            
            print("Itteration:",n)
            
            self.outputs = self.pcnfwd(inputs);
            self.weights += eta*py.dot(py.transpose(inputs),targets-self.outputs)
        
            # Randomise order of inputs
            #py.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
            print(self.weights);
            
        return self.weights

    def pcnfwd(self,inputs):

        outputs = py.dot(inputs,self.weights)

        # Threshold the outputs
        return py.where(outputs>0,1,0)


    def confmat(self,inputs,targets):

        # Add the inputs that match the bias node
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
        
    def logic(self):

        a = py.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
        b = py.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

        p = self.pcn(a[:,0:2],a[:,2:])
        p.pcntrain(a[:,0:2],a[:,2:],0.25,10)
        p.confmat(a[:,0:2],a[:,2:])

        q = self.pcn(a[:,0:2],b[:,2:])
        q.pcntrain(a[:,0:2],b[:,2:],0.25,10)
        q.confmat(a[:,0:2],b[:,2:])
