# -*- coding: utf-8 -*-
# psyco optimization -- WOW this makes it 2x as fast!
#import psyco
#psyco.full()

from Utilities import *
from LOTlib.EnumerativeSearch import *

show_baseline_distribution()
print "\n\n"

def random_lexicon():
    ret = GriceanSimpleLexicon(grammar, args=['A', 'B', 'S'])
    for w in target.all_words(): ret.set_word(w, grammar.generate('START'))
    return ret


# intialize a learner lexicon, at random
#learner = random_lexicon()
#for w in target.all_words():
    #learner.set_word(w, grammar.generate('START')) # eahc word returns a true, false, or undef (None)

## sample the target data
data = generate_data(1500)

nt_moves = dict() # hash from nonterminals to a *list* of possible moves
for nt in grammar.nonterminals():

    fs = UniquePriorityQueue(N=100, max=True)
    for i in xrange(50000):
        g = grammar.generate(nt)
        fs.push(g, g.log_probability())

    nt_moves[nt] = fs.get_all()


## Update the target with the data
target.compute_likelihood(data)

cache = dict()

def next_states(L):
    """
            All possible trees we can get from t
            We loop over subnodes, replacing
    """
    for w in L.all_words():
        t = L.dexpr[w]

        # now try replacing all other nodes
        for tt in t:

            # yeild everything of the same type
            if tt.returntype == t.returntype:
                yield L.copy()

            for i in xrange(len(tt.args)):

                old_a = tt.args[i]

                if isinstance(old_a, FunctionNode):
                    for new_a in nt_moves[old_a.returntype]:
                        tt.args[i] = new_a
                        yield L.copy() # we go down and copy t and the new node

                    tt.args[i] = old_a

def score(L): return sum(L.compute_posterior(data))

print "*** Target likelihood: ", target.compute_likelihood(data)


## Now we can enumerate!
start = [ random_lexicon() for i in xrange(10) ]


print "Done generating start states."


for l, s in enumerative_search( start , next_states, score, N=100000, breakout=1000):
    print s, l.compute_prior(), l.compute_likelihood(data)
    print l
