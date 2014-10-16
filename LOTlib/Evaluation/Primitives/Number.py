from Primitives import LOTlib_primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# counting list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from collections import defaultdict

# the next word in the list -- we'll implement these as a hash table
word_list = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']
next_hash, prev_hash = [defaultdict(lambda: 'undef'), defaultdict(lambda: 'undef')]
for i in range(1, len(word_list)-1):
    next_hash[word_list[i]] = word_list[i+1]
    prev_hash[word_list[i]] = word_list[i-1]
next_hash['one_'] = 'two_'
next_hash['ten_'] = 'undef'
prev_hash['one_'] = 'undef'
prev_hash['ten_'] = 'nine_'
next_hash['X'] = 'X'
prev_hash['X'] = 'X'

word_to_number = dict() # map a word like 'four_' to its number, 4
for i in range(len(word_list)):
    word_to_number[word_list[i]] = i+1
word_to_number['ten_'] = 'A' # so everything is one character
word_to_number['undef'] = 'U'

prev_hash[None] = None

@LOTlib_primitive
def next_(w): return next_hash[w]

@LOTlib_primitive
def prev_(w): return prev_hash[w]

@LOTlib_primitive
def ifU_(C,X):
    if C:
        return X
    else:
        return 'undef'
