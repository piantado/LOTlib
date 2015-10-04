"""
TODO: Allow a way to output to go to stderr
"""

from SampleStream import SampleStream
from sys import stdout

class Print(SampleStream):
    def __init__(self, file=None, prefix='', mode='w'):
        self.__dict__.update(locals())
        SampleStream.__init__(self, generator=None)

    def process(self, x):
        # print "Print.process ", x
        print >>self.file_, self.prefix, x
        return x

    def __enter__(self):
        SampleStream.__enter__(self)

        if self.file is not None:
            self.file_ = open(self.file, self.mode)
        else:
            self.file_ = stdout


    def __exit__(self, t, value, traceback):
        if self.file is not None:
            self.file_.close()

        SampleStream.__exit__(self, t, value, traceback)

