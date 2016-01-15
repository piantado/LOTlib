from collections import Counter
from FormalLanguage import FormalLanguage


class Dyck(FormalLanguage):
    """
        This one is very hard to learn, please run it with at least 1e5 MCMC steps
    """
    def __init__(self, A='a', B='b', max_length=8):
        assert len(A) == 1 and len(B) == 1, 'atom length should be one'

        self.A = A
        self.B = B

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):
        """
        we iterate over the space using dynamic programming, complexity is O(n^2*L^3), maybe some improvement?
        """
        assert max_length % 2 == 0, 'length should be even'
        memo = Counter([''])
        memo_new = Counter()
        for _ in xrange(max_length/2):
            for e in memo:
                s_len = len(e)
                for i in xrange(s_len+1):
                    for j in xrange(i+1, s_len+2):
                        s = Dyck.insert_str(Dyck.insert_str(e, self.A, i), self.B, j)
                        if self.is_valid_string(s) and not memo_new.has_key(s):
                            memo_new[s] += 1
                            yield s
            memo = memo_new; memo_new = Counter()

    @staticmethod
    def insert_str(s, c, p):
        """
        insert char c into s at position p
        NOTE: not sure it works with negative p !
        """
        s_len = len(s)
        assert 0 <= p <= s_len

        return s[:p] + c + s[p:]

    def string_log_probability(self, s):
        return -len(s)/2

    def is_valid_string(self, s):

        if len(s) < 2: return False

        cnt = 0
        for e in s:
            if e == self.A:
                cnt+=1
            elif e==self.B:
                cnt-=1
                if cnt < 0: return False

        return cnt == 0

# just for testing
if __name__ == '__main__':

    # print Dyck.insert_str('', '(', 0)
    #
    # print Dyck.insert_str('()', '(', 0)
    # print Dyck.insert_str('()', '(', 1)
    # print Dyck.insert_str('()', '(', 2)

    lang = Dyck()

    for e in lang.all_strings(max_length=8):
        print e

    # language = AnBn()
    #
    # for e in language.all_strings(max_length=20):
    #     print e
    #
    # print language.sample_data_as_FuncData(128, max_length=20)
    #
    # print language.is_valid_string('aaa')
    # print language.is_valid_string('ab')
    # print language.is_valid_string('abb')
    # print language.is_valid_string('aaab')
    # print language.is_valid_string('aabb')