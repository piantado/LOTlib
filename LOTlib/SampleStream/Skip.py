from SampleStream import SampleStream

class Skip(SampleStream):
    def __init__(self, n=10):
        SampleStream.__init__(self)
        self.n = n
        self.cnt = 0

    def process(self, x):
        self.cnt += 1
        if self.cnt % self.n == 0:
            return x
        else:
            return None
