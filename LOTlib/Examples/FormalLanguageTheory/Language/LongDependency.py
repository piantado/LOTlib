from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
import itertools


class LongDependency(FormalLanguage):

    def __init__(self, A='a', B='b', max_length=5):
        """
        NOTE: we specify the size of pool from which we will draw our X in sampling strings instead of max_length
        """
        assert len(A) == 1 and len(B) == 1, 'atom length should be one'

        self.A = A
        self.B = B
        self.C = ['c', 'd', 'e', 'f']

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        assert max_length > 1, 'pool_size should be larger than 2'

        num = 0
        for e in list(itertools.product(*([self.C]*3))):
            num += 1
            if num > max_length: break
            yield self.A + self.t2s(e) + self.B

    def estimate_precision_and_recall(self, h, data):
        """
        Re-implement this function in order to investigate how our model learns the dependency, we only cares about the precision
        """
        output = self.A + self.B
        h_out = [self.ht(h()) for _ in xrange(int(sum(data[0].output.values())))]

        base = len(h_out)
        cnt = 0.0
        for v in h_out:
            if v == output: cnt += 1
        precision = cnt / base

        return precision, precision

    def ht(self, s):
        """
        get head and tail of s
        """
        if s is None or len(s) < 2: return None
        return s[0] + s[-1]

    def t2s(self, t):
        s = ''
        for e in t:
            s += e
        return s


# just for testing
if __name__ == '__main__':
    language = LongDependency(pool_size=12)

    for e in language.all_strings(max_length=12):
        print e

    print language.sample_data_as_FuncData(128)
    # print language.is_valid_string('aaa')
    # print language.is_valid_string('ab')
    # print language.is_valid_string('abb')
    # print language.is_valid_string('aaab')
    # print language.is_valid_string('aabb')