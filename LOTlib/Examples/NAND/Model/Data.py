from LOTlib.DataAndObjects import FunctionData, make_all_objects
from LOTlib.Miscellaneous import sample_one
from Grammar import SHAPES, COLORS


# ------------------------------------------------------------------
# Set up the objects
# ------------------------------------------------------------------

all_objects = make_all_objects( shape=SHAPES, color=COLORS )


# ------------------------------------------------------------------
# Generator for data
# ------------------------------------------------------------------

def generate_data(N, f):
    data = []
    for _ in xrange(N):
        o = sample_one(all_objects)
        data.append(FunctionData(input=[o], output=f(o)))
    return data
