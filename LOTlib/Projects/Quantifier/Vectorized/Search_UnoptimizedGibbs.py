# -*- coding: utf-8 -*-

"""
A gibbs sampler version, but this does not cache all of the hypothesis responses. So it's slower,
but uses less memory.

This uses LOTlib.MetropolisHastings gibbs sampler, which calls enumerative proposer to sample.

Note: Mainly for experimentation.

Here, we wrap a GriceanSimpleLexicon with an enumerative sampler for each word meaning

"""
import pickle
from ..Model import *

IN_PATH = "data/all_trees_2012May2.pkl"
STEPS = 10000
data_size = 350

# load the list of individual word meanings
inh = open(IN_PATH)
fs = pickle.load(inh)
my_finite_trees = fs.get_all()


class GibbsyGriceanSimpleLexicon(GriceanSimpleLexicon):
    """A kind of SimpleLexicon which can do word-wise gibbs sampling.

    Overwrite enumerative_proposer to take a word, and sub in all meanings from our finite list.

    """
    def enumerative_proposer(self, wd):
        for k in my_finite_trees:
            new = self.copy()
            new.set_word(wd, k)
            yield new
    def copy(self):
        """Overwritten to work correctly for this type!"""
        new = GibbsyGriceanSimpleLexicon(self.grammar, self.args, alpha=self.alpha, palpha=self.palpha)
        for w in self.lex.keys(): new.lex[w] = self.lex[w].copy()
        return new


################################################################
## Now just run

# initialize the data
data = generate_data(data_size)

# get all of the words
all_words = target.all_words()

# intialize a learner lexicon, at random
learner = GibbsyGriceanSimpleLexicon(grammar, args=['A', 'B', 'S'])

for w in all_words:
    learner.set_word(w, grammar.generate('START')) # eahc word returns a true, false, or undef (None)


for g in LOTlib.MetropolisHastings.gibbs_sample(learner, data, STEPS, dimensions=all_words):

    print g.lp, g.compute_likelihood(data), target.compute_likelihood(data), g
