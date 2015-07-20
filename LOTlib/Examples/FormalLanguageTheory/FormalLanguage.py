class FormalLanguage(object):
    """
    Set up a class for formal languages, so we can compute things like accuracy, precision, etc.
    """

    def __init__(self, max_length=10):
        self.max_length = max_length

    def all_strings(self):
        """ Return all strings up to length maxlength """
        pass

    def string_log_probability(self, s):
        """ What is the prior of s under my assumed language? """

    def is_valid_string(self, s):
        """ Returns True if s is a valid string in this language """
        pass

    def sample_data(self, n):
        """ Return a dictionary of {string:count}  that is a sample from this language
        """
        pass

    def estimate_precision_and_recall(self, h):

        output = set(self.sample_data(1000)[0].output.keys())
        h_out = set([h() for _ in xrange(1000)])

        base = len(h_out)
        cnt = 0.0
        for v in h_out:
            if v in output: cnt += 1
        precision = cnt / base

        base = len(output)
        cnt = 0.0
        for v in output:
            if v in h_out: cnt += 1
        recall = cnt / base

        return precision, recall

    def estimate_KL_divergence(self, h):
        """ Estimate the KL divergence between me and h """
        pass
