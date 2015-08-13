from SampleStream import SampleStream

class Skip(SampleStream):
    """
    Thin our samples
    """

    def __init__(self, generator=None, skip=10):
        self.__dict__.update(locals())

        self.skip_idx = 0

        SampleStream.__init__(self, generator=generator)

    def update(self, generator):
        for x in generator:
            self.skip_idx += 1

            if self.skip_idx % self.skip == 0:
                yield x

    def add(self, x):
        pass # Do nothing!
