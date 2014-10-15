from LOTlib.DataAndObjects import *
from LOTlib.Miscellaneous import *
from Global import all_objects

# ------------------------------------------------------------------
# Generator for data
# ------------------------------------------------------------------

def generate_data(N, f):
    data = []
    for _ in xrange(N):
        o = sample_one(all_objects)
        data.append( FunctionData(input=[o], output=f(o) ) )
    return data
