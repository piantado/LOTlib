
from SampleStream import SampleStream

class Delta(SampleStream):
    """
    Only show changes in the sample
    """
    def __init__(self, generator=None):
        self.last = None
        SampleStream.__init__(self, generator=generator)

    def process(self, x):
        if self.last is not None and x == self.last:
            return None
        else:
            self.last = x
            return x
