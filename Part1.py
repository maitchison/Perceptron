from numpy import *
import pcn

# Tasks:
# [1] Tidy up the code a bit, renaming stuff and making sensiable functions
# [2] Add support for other bitwise logic functions 
# [3] Graphicly show input data 
# [4] Graphicly show linear seperation
# [5] Try different functions such as synoid and half rectifier


class LogicFunctions:
    OR = array([[0],[1],[1],[1]])
    AND = array([[0],[0],[0],[1]])
    XOR = array([[0],[1],[1],[0]])
    

# Inputs for 'or' function. 
inputs = array([[0,0],[0,1],[1,0],[1,1]])
targets = array([[0],[1],[1],[1]])

targets = LogicFunctions.AND

inputs_bias = concatenate((-ones((shape(inputs)[0],1)),inputs),axis=1)

p = pcn.pcn(inputs,targets)
p.pcntrain(inputs,targets,0.25,6)

#p.pcnfwd(inputs_bias,weights)