
from LOTlib.DataAndObjects import FunctionData

data = []
for x in xrange(1, 10):
    data.append(FunctionData(input=['even', x], output=(x % 2 == 0)))
    data.append(FunctionData(input=['odd',  x], output=(x % 2 == 1)))

def make_data():
    return data