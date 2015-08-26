import re
from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class AnBn(FormalLanguage):

    def __init__(self, A='a', B='b', max_length=10):
        assert len(A) == 1 and len(B) == 1, 'atom length should be one'

        self.A = A
        self.B = B

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        assert max_length % 2 == 0, 'length should be even'

        for i in xrange(1, max_length/2+1):
            yield self.A * i + self.B * i

    def string_log_probability(self, s):
        return -len(s)/2


# just for testing
if __name__ == '__main__':
    language = AnBn()

    for e in language.all_strings(max_length=20):
        print e

    print language.sample_data_as_FuncData(128)

    print language.is_valid_string('aaa')
    print language.is_valid_string('ab')
    print language.is_valid_string('abb')
    print language.is_valid_string('aaab')
    print language.is_valid_string('aabb')