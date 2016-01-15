import re
from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class ABn(FormalLanguage):

    def __init__(self, atom='ab', max_length=10):
        assert max_length >= 2
        self.atom = atom
        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        for i in xrange(2, max_length+1, 2):
            yield self.atom * (i / 2)

    def string_log_probability(self, s):
        return -len(s)/2

# just for testing
if __name__ == '__main__':
    language = ABn()

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128)