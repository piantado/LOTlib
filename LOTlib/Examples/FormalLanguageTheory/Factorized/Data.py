
from LOTlib.DataAndObjects import FunctionData

def make_data(n=64):
    n = float(n)
    return [FunctionData(input=(), output={'abc': n,
                                           'aabbcc': n/2,
                                           'aaabbbccc': n/4,
                                           'aaaabbbbcccc': n/8,
                                           'aaaaabbbbbccccc': n/16,
                                           'aaaaaabbbbbbcccccc': n/32})]


# Another data set: Dyck language
# def make_data(n=64):
#     return [FunctionData(input=(), output={'()': n,
#                                            '(())': n/2,
#                                            '()()': n/2,
#                                            '((()))': n/4,
#                                            '(()())': n/4,
#                                            '()()()': n/4,
#                                         })]

# Check out some examples here:
# https://en.wikipedia.org/wiki/Indexed_language

# Or do the Bach language:
#https://en.wikipedia.org/wiki/Context-sensitive_language