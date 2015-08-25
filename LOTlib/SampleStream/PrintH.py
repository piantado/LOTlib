## TODO: In __init__, we need to set the encoding

from Print import Print
from LOTlib.Miscellaneous import qq
from LOTlib.FunctionNode import cleanFunctionNodeString

class PrintH(Print):
    """ Fancier printing for hypotheses """
    def __init__(self, *args, **kwargs):

        Print.__init__(self, *args, **kwargs)



    def process_(self, x):
        if self.prefix is not None:
            print >>self.file_, self.prefix,

        print >>self.file_, \
              round(x.posterior_score,3), \
              round(x.prior,3), \
              round(x.likelihood,3), \
              qq(x)
              # qq(cleanFunctionNodeString(x))
        return x