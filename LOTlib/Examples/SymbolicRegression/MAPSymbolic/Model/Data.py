"""
Data generation for symbolic regression
"""

from random import random
from numpy.random import normal
from numpy import sin

from LOTlib.DataAndObjects import FunctionData

def F(x):
    return 3.*x + sin(4.3/x)

def make_data(target=F, data_size=100, sd=1.0):
    """Make up some learning data for the symbolic regression."""

    data = []
    for i in range(data_size):
        x = random()
        y = target(x) + normal()*sd
        data.append( FunctionData(input=[x], output=y, ll_sd=sd) )

    return data
