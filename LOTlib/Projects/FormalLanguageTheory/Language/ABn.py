import re
from LOTlib.Projects.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class ABn(FormalLanguage):

    def __init__(self, atom='ab', max_length=20):
        assert max_length % 2 == 0 and max_length >= 2, 'invalid max_length'
        self.atom = atom
        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        for i in xrange(1, max_length/2 + 1):
            yield self.atom * i

    def string_log_probability(self, s):
        return -len(s)/2


# just for testing
if __name__ == '__main__':
    language = ABn()

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128)