
from SampleStream import SampleStream

class Unique(SampleStream):
    """
    Only return unique samples
    """

    def __init__(self, generator=None):
        self.seen = set()

        SampleStream.__init__(self, generator)

    def update(self, generator):
        for x in generator:
            if x not in self.seen:
                self.seen.add(x)
                yield x