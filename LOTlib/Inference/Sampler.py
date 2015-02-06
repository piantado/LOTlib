

class Sampler(object):
    """
    An abstract sampler class
    """

    def __init__(self):
        raise NotImplementedError

    def get_state(self):
        return self.current_sample

    def set_state(self, s, compute_posterior=True):
        self.current_sample = s
        if compute_posterior:
            self.current_sample.compute_posterior(self.data)

    def str(self):
        return "<%s sampler in state %s>" % (type(self), self.current_sample)

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError

    def take(self, n, **kwargs):
        """
        The next n samples
        """
        for _ in xrange(n):
            yield self.next(**kwargs)

    def compute_posterior(self, h, data):
        """
                A wrapper for hypothesis.compute_posterior(data) that can be overwritten in subclassses for fanciness
                Should return [np,nl], the prior and likelihood
        """
        return h.compute_posterior(data)