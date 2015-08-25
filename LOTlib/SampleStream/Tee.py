from SampleStream import SampleStream

class Tee(SampleStream):
    """ Split a sample. Returns (via yield) only the first """
    def __init__(self, *children):
        SampleStream.__init__(self)
        self.children = children

    def process(self, x):
        ret = None
        for i, c in enumerate(self.children):
            if i == 0:
               ret = c.process(x)
            else:
                c.process(x)
        return ret

