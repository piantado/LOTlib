from LOTlib import lot_iter

class MCMCSummary:
    """Superclass for collecting, computing, and displaying summary statistics from model runs.

    Arguments:
        skip (int): Skip this many samples before adding another; e.g. if skip=100, take 1 sample every 100.
        cap (int): If we have collected this many samples, collect no more.

    Attributes:
        samples (list): List of samples collected during run.
        sample_count (int): Number of samples collected.
        count (int): Number of samples taken as input to `add`; this is different from `sample_count`.

    Example:
        >> summary = InferenceSummary(skip=1000, cap=100)

        # Can be used either as an independent object...
        for h in lot_iter(mh_sampler):
            summary.add(h)

        # Or as a generator...
        for h in summary(mh_sampler):
            do_other_stuff(h)

    """
    def __init__(self, skip=100, cap=100):
        self.skip = skip
        self.cap = cap
        self.samples = []
        self.sample_count = 0
        self.count = 0
        self.top_samples = None

    def add(self, sample):
        """Append another sample to `self.samples` for every s input items we get (where s is `self.skip`).

        Do not add the sample if we already have at least `self.cap` samples.

        """
        if (self.count % self.skip == 0) and (self.sample_count < self.cap):
            self.samples.append(sample)
            self.sample_count += 1
        self.count += 1

    def __call__(self, generator):
        """Pass this a generator, add each element as it's yielded; allows us to make a pipeline."""
        for sample in lot_iter(generator):
            self.add(sample)
            yield sample

    def print_top_samples(self):
        if self.top_samples is None:
            self.set_top_samples()
        print '~'*100, '\nTop Samples:'
        for s in self.top_samples:
            print '*'*90
            print 'Value: ', s.value
            print 'Prior: %.3f' % s.prior, '\tLikelihood: %.3f' % s.likelihood, \
                '\tPostScore: %.3f' % s.posterior_score

    def set_top_samples(self, n=10, key=lambda x: x.posterior_score):
        self.top_samples = sorted(self.samples, key=key)[-n:]
        return self.top_samples
