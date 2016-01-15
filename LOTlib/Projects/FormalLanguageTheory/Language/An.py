import re
from FormalLanguage import FormalLanguage


class An(FormalLanguage):

    def __init__(self, atom='a', max_length=10):
        self.atom = atom
        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        for i in xrange(1, max_length+1):
            yield self.atom * i


# just for testing
if __name__ == '__main__':
    language = An()

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128)