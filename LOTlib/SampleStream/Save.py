import pickle
from SampleStream import SampleStream

class Save(SampleStream):
    def __init__(self, path='samples.pkl'):
        SampleStream.__init__(self, generator=None)

        self.path = path
        self.samples = []

    def process(self, x):
        self.samples.append(x)
        return x

    def __exit__(self, t, value, traceback):

        f = open(self.path, 'w')
        pickle.dump(self.samples, f)
        f.close()

        SampleStream.__exit__(self, t, value, traceback)