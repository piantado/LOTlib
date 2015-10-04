from SampleStream import SampleStream

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

    def process(self, h):
        self.posteriors.append(  getattr(h, 'posterior_score') )
        self.priors.append(      getattr(h, 'prior') )
        self.likelihoods.append( getattr(h, 'likelihood') )

        n = len(self.posteriors)
        if n>0 and self.plot_every is not None and n % self.plot_every == 0:
            self.plot()

        return h

    def plot(self):
        import matplotlib.pyplot as plt
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

    def __exit__(self, t, value, traceback):
        self.plot()

        return SampleStream.__exit__(self, t, value, traceback)