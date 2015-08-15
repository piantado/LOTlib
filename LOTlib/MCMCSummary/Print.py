"""
TODO: Allow a way to output to go to stderr
"""

from SampleStream import SampleStream
from sys import stdout

class Print(SampleStream):
    def __init__(self, file=None, prefix=None, mode='w'):
        self.__dict__.update(locals())
        SampleStream.__init__(self, generator=None)

        if self.file is not None:
            self.file_ = open(self.file, self.mode)
        else:
            self.file_ = stdout


    def process_(self, x):
        if self.prefix is not None:
            print >>self.file_, self.prefix,
        print >>self.file_, x
        return x

    def __exit__(self, t, value, traceback):
        if self.file is not None:
            self.file_.close()

        SampleStream.__exit__(self, t, value, traceback)

