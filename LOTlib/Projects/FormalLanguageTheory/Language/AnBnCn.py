import re
from FormalLanguage import FormalLanguage


class AnBnCn(FormalLanguage):

    def __init__(self, A='a', B='b', C='c', max_length=12):
        assert len(A) == 1 and len(B) == 1 and len(C) == 1, 'atom length should be one'

        self.A = A
        self.B = B
        self.C = C

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):

        assert max_length % 3 == 0, 'length should be divisible by 3'

        for i in xrange(1, max_length/3+1):
            yield self.A * i + self.B * i + self.C * i

    def string_log_probability(self, s):
        return -len(s)/3


# just for testing
if __name__ == '__main__':
    language = AnBnCn()

    for e in language.all_strings(max_length=30):
        print e

    print language.sample_data_as_FuncData(128)
    print language.is_valid_string('aaabb')
    print language.is_valid_string('abc')
    print language.is_valid_string('abbcc')
    print language.is_valid_string('aaabc')
    print language.is_valid_string('aabbcc')