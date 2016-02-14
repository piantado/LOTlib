from LOTlib.Examples.SymbolicRegression import make_hypothesis

## Here we just overwrite make_data

from math import sin
from random import random
from numpy.random import normal
from LOTlib.DataAndObjects import FunctionData

## The target function for symbolic regression
def F1(x):
    return x + sin(1.0/x)

# Make up some learning data for the symbolic regression
def make_data(n=1, target=F1, data_size=100, sd=0.1):

    # initialize the data
    data = []
    for i in range(data_size):
        x = random()
        y = target(x) + normal()*sd
        data.append( FunctionData(input=[x], output=y, ll_sd=sd) )

    return data*n
