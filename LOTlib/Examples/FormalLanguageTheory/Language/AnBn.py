import re
from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class AnBn(FormalLanguage):

    def __init__(self, A='a', B='b'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 1, 'atom length should be one'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B

    def all_strings(self, max_length=50):

        assert max_length % 2 == 0, 'length should be even'

        for i in xrange(1, max_length/2+1):
            yield self.A * i + self.B * i

    def is_valid_string(self, s):
        re_atom = r'({}*)({}*)'.format(self.A, self.B)

        m = re.match(re_atom, s)
        if m:
            am, bm = m.groups()
            return len(am) == len(bm)
        else:
            return False

    def string_log_probability(self, s):
        return -len(s)/2


# just for testing
if __name__ == '__main__':
    language = AnBn()

    for e in language.all_strings(max_length=20):
        print e

    print language.sample_data_as_FuncData(128, max_length=20)

    print language.is_valid_string('aaa')
    print language.is_valid_string('ab')
    print language.is_valid_string('abb')
    print language.is_valid_string('aaab')
    print language.is_valid_string('aabb')