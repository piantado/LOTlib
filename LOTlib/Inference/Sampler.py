

class Sampler(object):
    """
    Sampler class template. Generator format, call __iter__() or next() to yield more samples.

    'States' for the sampler refer to the most recently yielded sample.

    """

    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError

    def get_state(self):
        return self.current_sample

    def set_state(self, s, compute_posterior=True):
        """Set the current sample, maybe compute its posterior.

        Args
        ----
        s : Hypothesis
            Set `self.current_sample` to this.
        compute_posterior : bool
            If true, compute its posterior.

        """
        self.current_sample = s
        if compute_posterior:
            self.current_sample.compute_posterior(self.data)

    def str(self):
        return "<%s sampler in state %s>" % (type(self), self.current_sample)

    def take(self, n, **kwargs):
        """
        Yield the next `n` samples.

        """
        for _ in xrange(n):
            yield self.next(**kwargs)

    def compute_posterior(self, h, data):
        """
        A wrapper for hypothesis.compute_posterior(data) that can be overwritten in fancy subclassses.

        Should return [np,nl], the prior and likelihood

        """
        return h.compute_posterior(data)