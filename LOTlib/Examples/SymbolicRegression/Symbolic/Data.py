from math import sin
import random
from LOTlib.DataAndObjects import FunctionData

data_sd = 0.10 # the SD of the likelihood

## The target function for symbolic regression
target = lambda x: x + sin(1.0/x)

# Make up some learning data for the symbolic regression
def generate_data(data_size):

    # initialize the data
    data = []
    for i in range(data_size):
        x = random.random()
        data.append( FunctionData(input=[x], output=target(x), ll_sd=data_sd) )

    return data
