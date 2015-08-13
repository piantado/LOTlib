from LOTlib.Miscellaneous import qq
from SampleStream import SampleStream
from LOTlib.FunctionNode import cleanFunctionNodeString

class Print(SampleStream):
    """
    Display samples in a standardized format
    """

    def add(self, x):
        print round(x.posterior_score,3), \
              round(x.prior,3), \
              round(x.likelihood,3), \
              qq(cleanFunctionNodeString(x))
