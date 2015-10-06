
import pickle
from LOTlib import break_ctrlc


class MCMCSummary(object):
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
        >> for h in break_ctrlc(mh_sampler):
        >>     summary.add(h)

        # Or as a generator...
        >> for h in summary(mh_sampler):
        >>     do_other_stuff(h)

    """
    def __init__(self, skip=100, cap=100):
        self.skip = skip
        self.cap = cap
        self.samples = []
        self.sample_count = 0
        self.count = 0

    def __call__(self, generator):
        """Pass this a generator, add each element as it's yielded.

        This allows us to make a pipeline. See Example in main docstring: '# Or as a generator...'.

        """
        if hasattr(generator, 'data'):
            self.data = generator.data
        for sample in break_ctrlc(generator):
            self.add(sample)
            yield sample

    def add(self, sample):
        """
        Append a sample to `self.samples`, once every `self.skip` calls, until we reach `self.cap`.

        """
        if (self.count % self.skip == 0) and (self.sample_count < self.cap):
            self.samples.append(sample)
            self.sample_count += 1
        self.count += 1

    def pickle_summary(self, filename='MCMC_summary_data.p'):
        f = open(filename, "wb")
        pickle.dump(self, f)

    # --------------------------------------------------------------------------------------------------------
    # Top samples in `self.samples`

    def get_top_samples(self, n=10, s_idx=None, key=(lambda x: x.posterior_score)):
        """Get the top `n` GrammarHypothesis samples in `self.samples`, sorted by specified key.

        Args:
            n (int): Get the top `n` samples.
            s_idx (int): We only consider samples 1 through this one. E.g. if idx = 2, we only consider the
              first 3 samples.
            key (function): Lambda function, this tells us how to sort our samples to get the top `n`.

        """
        if s_idx is None:        # Consider only the samples with indexes specified by `idxs`
            s_idx = range(0, self.sample_count)
        samples = [self.samples[i] for i in s_idx]
        sorted_samples = sorted(samples, key=key)
        sorted_samples.reverse()
        return sorted_samples[-n:]

    def print_top_samples(self, **kwargs):
        top_samples = self.get_top_samples(**kwargs)
        print '~'*100, '\nTop Samples:'
        for s in top_samples:
            print '*'*90
            print 'Value: ', s.value
            print 'Prior: %.3f' % s.prior, '\tLikelihood: %.3f' % s.likelihood, \
                '\tPostScore: %.3f' % s.posterior_score

    # --------------------------------------------------------------------------------------------------------
    # Top hypotheses in `self.hypotheses`

    def get_top_hypotheses(self, n=10, idx=None, gh_key='recent', h_key=(lambda x: x.posterior_score)):
        """Get the `n` top hypotheses, not GrammarHypotheses but the ones stored in `self.hypotheses`.

        Using the `gh_key`, we pick just 1 GrammarHypothesis sample, and we get the top hypotheses
          according to that. This means we can see the different top hypotheses for the MLE,
          MAP, or mean GrammarHypothesis

        Args:
            n (int): Number of top hypotheses to get.
            gh_key (str): We get the top `self.hypotheses` for this GrammarHypothesis.
            h_key (function): Lambda function, this tells us how to sort `gh.hypotheses`.

        """
        if idx is None:
            idx = self.sample_count
        sample_idxs = range(0, idx)

        if gh_key is 'recent':      # Most recent sample
            gh = self.samples[idx]
        if gh_key is 'MLE':         # Most likely GH
            gh = self.get_top_samples(n=1, s_idx=sample_idxs, key=(lambda x: x.likelihood))[0]
        if gh_key is 'MAP':         # Max Post. GH  (should prob be very close to MLE
            gh = self.get_top_samples(n=1, s_idx=sample_idxs, key=(lambda x: x.posterior_score))[0]
        if gh_key is 'mean':        # Mean GH (this should create a new one... right?
            mean_value = self.mean_value(sample_idxs)
            gh = self.samples[-1].__copy__()
            gh.set_value(mean_value)

        sorted_hypotheses = sorted(gh.hypotheses, key=h_key)
        sorted_hypotheses.reverse()
        return sorted_hypotheses[-n:]

    def print_top_hypotheses(self, n=10, idx=None, gh_key='mean'):
        """Print the top hypotheses for the `gh_key` GrammarHypothesis specified, for samples[0:idx]."""
        top_hypotheses = self.get_top_hypotheses(n=n, idx=idx, gh_key=gh_key)
        print '='*100, '\nTop Hypotheses for ', gh_key, '[0:', idx, '] GrammarHypothesis:'
        for h in top_hypotheses:
            print '*'*90
            print 'Hypothesis: ', ['%.3f' % str(h)]
            print 'Prior: %.3f' % h.prior, '\tLikelihood: %.3f' % h.likelihood, \
                  '\tPosterior: %.3f' % h.posterior_score




