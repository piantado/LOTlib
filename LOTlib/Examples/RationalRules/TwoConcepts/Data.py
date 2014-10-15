# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We need data in a different format for this guy

from LOTlib.DataAndObjects import *

# The argumetns are [concept,object]
data = [
    FunctionData( ['A', Obj(shape='square', color='red')],    True), \
    FunctionData( ['A', Obj(shape='square', color='blue')],   False), \
    FunctionData( ['A', Obj(shape='triangle', color='blue')], False), \
    FunctionData( ['A', Obj(shape='triangle', color='red')],  False), \

    FunctionData( ['B', Obj(shape='square', color='red')],    False), \
    FunctionData( ['B', Obj(shape='square', color='blue')],   True), \
    FunctionData( ['B', Obj(shape='triangle', color='blue')], True), \
    FunctionData( ['B', Obj(shape='triangle', color='red')],  True)
] * 10  # number of data points exactly like these
