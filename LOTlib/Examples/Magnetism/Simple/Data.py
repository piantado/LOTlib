from LOTlib.DataAndObjects import FunctionData


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up data -- true output means attraction (p=positive; n=negative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = [ FunctionData(input=[ "p1", "n1" ], output=True),
                 FunctionData(input=[ "p1", "n2" ], output=True),
                 FunctionData(input=[ "p1", "p1" ], output=False),
                 FunctionData(input=[ "p1", "p2" ], output=False),

                 FunctionData(input=[ "p2", "n1" ], output=True),
                 FunctionData(input=[ "p2", "n2" ], output=True),
                 FunctionData(input=[ "p2", "p1" ], output=False),
                 FunctionData(input=[ "p2", "p2" ], output=False),

                 FunctionData(input=[ "n1", "n1" ], output=False),
                 FunctionData(input=[ "n1", "n2" ], output=False),
                 FunctionData(input=[ "n1", "p1" ], output=True),
                 FunctionData(input=[ "n1", "p2" ], output=True),

                 FunctionData(input=[ "n2", "n1" ], output=False),
                 FunctionData(input=[ "n2", "n2" ], output=False),
                 FunctionData(input=[ "n2", "p1" ], output=True),
                 FunctionData(input=[ "n2", "p2" ], output=True)]

def make_data(*args, **kwargs):
    return data

