from SampleStream import SampleStream

class Tee(SampleStream):
    """ Split a sample. Returns (via yield) only the first """
    def __init__(self, *outputs):
        SampleStream.__init__(self)

        ##UUGH because of the weird associativity, >> will return the last in the chain.
        ## But really my child should be the root of whatever chain I got.
        self.outputs = [o.root() for o in outputs]
