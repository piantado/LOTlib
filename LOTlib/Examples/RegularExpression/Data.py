from LOTlib.DataAndObjects import FunctionData

##########################################################
# Define some data

data = [
    FunctionData(input=['aaaa'], output=True),
    FunctionData(input=['aaab'], output=False),
    FunctionData(input=['aabb'], output=False),
    FunctionData(input=['aaba'], output=False),
    FunctionData(input=['aca'],  output=True),
    FunctionData(input=['aaca'], output=True),
    FunctionData(input=['a'],    output=True)
]
