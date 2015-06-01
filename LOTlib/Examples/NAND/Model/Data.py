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

# Just import some defaults
from LOTlib.Examples.NAND.TargetConcepts import TargetConcepts

def make_data(N=20, f=TargetConcepts[0]):
    data = []
    for _ in xrange(N):
        o = sample_one(all_objects)
        data.append(FunctionData(input=[o], output=f(o)))
    return data
