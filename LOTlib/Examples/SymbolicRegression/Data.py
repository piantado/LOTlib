"""

    Data generation for symbolic regression

"""

from random import random
from numpy.random import normal

from LOTlib.DataAndObjects import FunctionData

# Make up some learning data for the symbolic regression
def generate_data(target, data_size, sd):

    # initialize the data
    data = []
    for i in range(data_size):
        x = random()
        y = target(x) + normal()*sd
        data.append( FunctionData(input=[x], output=y, ll_sd=sd) )

    return data
