from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
import itertools


class LongDependency(FormalLanguage):

    def __init__(self, A='a', B='b', C='c'):
        """
        C can be list, in which case we will randomly pick the element in list when generate strings

        NOTE: for length l, number of combinations is len(C)^(l-2), so be sure to limit the length or len(C), in case getting too slow
        """
        assert len(A) == 1 and len(B) == 1, 'atom length should be one'

        if isinstance(C, list):
            for e in C: assert len(e) == 1, 'atom length should be one'
        else:
            C = [C]

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B
        self.C = C

    def all_strings(self, max_length=50):

        assert max_length > 2, 'length should be longer than 2'

        for i in xrange(3, max_length+1):
            for e in list(itertools.product(*([self.C]*(i-2)))):
                yield self.A + self.t2s(e) + self.B

    def is_valid_string(self, s):

        if len(s) <= 2:
            return False

        if s[0] != self.A or s[-1] != self.B:
            return False

        for i in xrange(1, len(s) - 1):
            if s[i] not in self.C:
                return False

        return True

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

        return precision, 1.0

    def ht(self, s):
        """
        get head and tail of s
        """
        return s[0] + s[-1]

    def t2s(self, t):
        s = ''
        for e in t:
            s += e
        return s


# just for testing
if __name__ == '__main__':
    language = LongDependency(C=['c', 'd', 'e'])

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128, max_length=10)

    print language.is_valid_string('aaa')
    print language.is_valid_string('ab')
    print language.is_valid_string('abb')
    print language.is_valid_string('aaab')
    print language.is_valid_string('aabb')