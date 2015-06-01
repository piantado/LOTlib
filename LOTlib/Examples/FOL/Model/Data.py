
from LOTlib.DataAndObjects import FunctionData, Obj # for nicely managing data

# Make up some data -- here just one set containing {red, red, green} colors that is mapped to True
data = [ FunctionData(input=[ {Obj(color='red'), Obj(color='red'), Obj(color='green')} ], output=True) ]

def make_data():
    return data
