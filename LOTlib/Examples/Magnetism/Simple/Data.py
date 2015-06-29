from LOTlib.DataAndObjects import FunctionData

alpha=0.99

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up data -- true output means attraction (p=positive; n=negative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = [ FunctionData(input=[ "p1", "n1" ], output=True, alpha=alpha),
                 FunctionData(input=[ "p1", "n2" ], output=True, alpha=alpha),
                 FunctionData(input=[ "p1", "p1" ], output=False, alpha=alpha),
                 FunctionData(input=[ "p1", "p2" ], output=False, alpha=alpha),

                 FunctionData(input=[ "p2", "n1" ], output=True, alpha=alpha),
                 FunctionData(input=[ "p2", "n2" ], output=True, alpha=alpha),
                 FunctionData(input=[ "p2", "p1" ], output=False, alpha=alpha),
                 FunctionData(input=[ "p2", "p2" ], output=False, alpha=alpha),

                 FunctionData(input=[ "n1", "n1" ], output=False, alpha=alpha),
                 FunctionData(input=[ "n1", "n2" ], output=False, alpha=alpha),
                 FunctionData(input=[ "n1", "p1" ], output=True, alpha=alpha),
                 FunctionData(input=[ "n1", "p2" ], output=True, alpha=alpha),

                 FunctionData(input=[ "n2", "n1" ], output=False, alpha=alpha),
                 FunctionData(input=[ "n2", "n2" ], output=False, alpha=alpha),
                 FunctionData(input=[ "n2", "p1" ], output=True, alpha=alpha),
                 FunctionData(input=[ "n2", "p2" ], output=True, alpha=alpha)]

def make_data(*args, **kwargs):
    return data

