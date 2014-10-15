from LOTlib.DataAndObjects import FunctionData, Obj

# # # # # # # # # # # # # # # # # # # # # #
# Make up some data - Let's give data from a simple conjunction (note this example data is not exhaustive)

# FunctionData takes a list of arguments and a return value. The arguments are objects (which are handled correctly
# automatically by is_color_ and is_shape_
data = [
    FunctionData( [Obj(shape='square', color='red')], True),
    FunctionData( [Obj(shape='square', color='blue')], False),
    FunctionData( [Obj(shape='triangle', color='blue')], False),
    FunctionData( [Obj(shape='triangle', color='red')], False),
 ]