import re
from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class An(FormalLanguage):

    def __init__(self, atom='a'):
        """
        don't use char like | and ) currently
        """
        FormalLanguage.__init__(self)
        self.atom = atom

    def all_strings(self, max_length=50):

        for i in xrange(1, max_length+1):
            yield self.atom * i

    def is_valid_string(self, s, max_length=50):

        re_atom = r'{}*'.format(self.atom)

        if re.match(re_atom, s):
            return len(s) <= max_length
        else:
            return False


# just for testing
if __name__ == '__main__':
    language = An()

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128, max_length=10)