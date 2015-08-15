
from SampleStream import SampleStream

class Unique(SampleStream):
    def __init__(self, generator=None):
        self.seen = set()
        SampleStream.__init__(self, generator=generator)

    def process_(self, x):
        if x in self.seen:
            return None
        else:
            self.seen.add(x)
            return x