import itertools
from LOTlib.DataAndObjects import FunctionData
from Grammar import objects

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up data -- true output means attraction (p=positive; n=negative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = []

for a,b in itertools.product(objects, objects):

    myinput  = [a,b]

    # opposites (n/p) interact; x interacts with nothing
    myoutput = (a[0] != b[0]) and (a[0] != 'x') and (b[0] != 'x')

    data.append( FunctionData(input=myinput, output=myoutput) )

def make_data():
    return data