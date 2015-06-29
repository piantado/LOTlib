
"""
Define some data.

"""
from LOTlib.DataAndObjects import FunctionData

alpha=0.99

data = [
    FunctionData(input=['aaaa'], output=True, alpha=alpha),
    FunctionData(input=['aaab'], output=False, alpha=alpha),
    FunctionData(input=['aabb'], output=False, alpha=alpha),
    FunctionData(input=['aaba'], output=False, alpha=alpha),
    FunctionData(input=['aca'],  output=True, alpha=alpha),
    FunctionData(input=['aaca'], output=True, alpha=alpha),
    FunctionData(input=['a'],    output=True, alpha=alpha)
]

def make_data(size=1):
    return data * size