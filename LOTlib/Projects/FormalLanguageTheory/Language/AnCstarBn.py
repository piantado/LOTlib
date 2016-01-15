import re
from FormalLanguage import FormalLanguage


class AnCstarBn(FormalLanguage):

    def __init__(self, A='a', B='b', C='c', max_length=12):
        assert len(A) == 1 and len(B) == 1 and len(C) == 1, 'atom length should be one'

        self.A = A
        self.B = B
        self.C = C

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        assert max_length % 2 == 0, 'length should be even'

        for i in xrange(1, max_length/2+1):
            for j in xrange(max_length - 2*i+1):
                yield self.A * i + self.C * j + self.B * i


# just for testing
if __name__ == '__main__':
    language = AnCstarBn()

    for e in language.all_strings(max_length=20):
        print e

    print language.sample_data_as_FuncData(128, max_length=20)

    print language.is_valid_string('aaac')
    print language.is_valid_string('acb')
    print language.is_valid_string('accbb')
    print language.is_valid_string('aaaccb')
    print language.is_valid_string('aaccbb')