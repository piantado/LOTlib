
from SampleStream import SampleStream

class Unique(SampleStream):
    def __init__(self):
        SampleStream.__init__(self)

        self.seen = set()

    def process(self, x):
        if x in self.seen:
            return None
        else:
            self.seen.add(x)
            return x