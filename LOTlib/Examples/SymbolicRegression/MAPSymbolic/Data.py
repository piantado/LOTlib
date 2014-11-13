"""
Data generation for symbolic regression

"""
from random import random
from numpy.random import normal

from LOTlib.DataAndObjects import FunctionData


def generate_data(target, data_size, sd):
    """Make up some learning data for the symbolic regression."""
    # initialize the data
    data = []
    for i in range(data_size):
        x = random()
        y = target(x) + normal()*sd
        data.append( FunctionData(input=[x], output=y, ll_sd=sd) )

    return data
