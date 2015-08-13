from SampleStream import SampleStream
import matplotlib.pyplot as plt

class PosteriorTrace(SampleStream):
    """
    A class for plotting/showing a posterior summary trace plot.

    """

    def __init__(self, generator=None, plot_every=1000, window=False, block=False, file='trace.pdf'):
        self.__dict__.update(locals())

        self.posteriors = []
        self.priors = []
        self.likelihoods = []

        SampleStream.__init__(self, generator)

    def add(self, h):
        self.posteriors.append(  getattr(h, 'posterior_score') )
        self.priors.append(      getattr(h, 'prior') )
        self.likelihoods.append( getattr(h, 'likelihood') )

        n = len(self.posteriors)
        if n>0 and n % self.plot_every == 0 :
            self.plot()

    def plot(self):

        plt.clf()
        plt.figure(2,figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.posteriors)
        plt.xlabel('# steps')
        plt.ylabel('posterior score')

        plt.subplot(1,2,2)
        plt.plot(self.priors, self.likelihoods)
        plt.xlabel('prior')
        plt.ylabel('likelihood')

        if self.file is not None:
            plt.savefig(self.file)

        if self.window:
            plt.show(block=self.block) # Apparently block is experimental

