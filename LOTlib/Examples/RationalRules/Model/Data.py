from LOTlib.DataAndObjects import FunctionData, Obj

# # # # # # # # # # # # # # # # # # # # # #
# Make up some data - Let's give data from a simple conjunction (note this example data is not exhaustive)

alpha = 0.99

# FunctionData takes a list of arguments and a return value. The arguments are objects (which are handled correctly
# automatically by is_color_ and is_shape_
data = [
    FunctionData( input=[Obj(shape='square', color='red')], output=True, alpha=alpha),
    FunctionData( input=[Obj(shape='square', color='blue')], output=False, alpha=alpha),
    FunctionData( input=[Obj(shape='triangle', color='blue')], output=False, alpha=alpha),
    FunctionData( input=[Obj(shape='triangle', color='red')], output=False, alpha=alpha),
 ]

def make_data():
    return data