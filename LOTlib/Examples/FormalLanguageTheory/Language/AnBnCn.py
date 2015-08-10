import re
from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage


class AnBnCn(FormalLanguage):

    def __init__(self, A='a', B='b', C='c'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 1 and len(C) == 1, 'atom length should be one'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B
        self.C = C

    def all_strings(self, max_length=50):

        assert max_length % 3 == 0, 'length should be divisible by 3'

        for i in xrange(1, max_length/3+1):
            yield self.A * i + self.B * i + self.C * i

    def is_valid_string(self, s):
        re_atom = r'({}*)({}*)({}*)'.format(self.A, self.B, self.C)

        m = re.match(re_atom, s)
        if m:
            am, bm, cm = m.groups()
            return len(am) == len(bm) == len(cm)
        else:
            return False

    def string_log_probability(self, s):
        return -len(s)/3


# just for testing
if __name__ == '__main__':
    language = AnBnCn()

    for e in language.all_strings(max_length=30):
        print e

    print language.sample_data_as_FuncData(128, max_length=30)

    print language.is_valid_string('aaabb')
    print language.is_valid_string('abc')
    print language.is_valid_string('abbcc')
    print language.is_valid_string('aaabc')
    print language.is_valid_string('aabbcc')