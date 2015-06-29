
from LOTlib.DataAndObjects import FunctionData

alpha = 0.99

data = []
for x in xrange(1, 10):
    data.append(FunctionData(input=['even', x], output=(x % 2 == 0), alpha=alpha))
    data.append(FunctionData(input=['odd',  x], output=(x % 2 == 1), alpha=alpha))

def make_data():
    return data