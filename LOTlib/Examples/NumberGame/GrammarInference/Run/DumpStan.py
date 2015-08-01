__author__ = 'Eric Bigelow'

# ============================================================================================================
# Setting up & dump data for Stan
# ===============================

from LOTlib.Examples.NumberGame.GrammarInference.Model import *
import numpy as np
import os, pickle

# Set up data
path = os.getcwd() + '/'
f1 = open(path + 'human_data.p')
data = pickle.load(f1)

# Set up GrammarHypothesis
ngh_file = path + 'out/ngh_lot300k1.p'
gh = NoConstGrammarHypothesis(lot_grammar, [], load=ngh_file)

# Initialize stuff
gh.init_C()
all_queries = set.union( *[set(d.queries) for d in data] )


# int<lower=1> h;                     // # of domain hypotheses
# int<lower=1> r;                     // # of rules in total
# int<lower=0> d;                     // # of data points
# int<lower=0> q;                     // max # of queries for a given datum

# int<lower=0> C[h,r];                // rule counts for each hypothesis
# real<upper=0> L[h,d];               // likelihood of data.input
# int<lower=0,upper=1> R[h,d,q];      // is each data.query in each hypothesis  (1/0)
# int<lower=0> D[d,q,2];              // human response for each data.query  (# yes, # no)

h = len(gh.hypotheses)
r = gh.n
d = len(data)
q = len(all_queries)

# vector<upper=0>[h] L[d];            // log likelihood of data.input
# vector<lower=0,upper=1>[h] R[d,q];  // is each data.query in each hypothesis  (1/0)

C = gh.C
# L = np.zeros((h, d))
L = [np.zeros((h))] * d
# R = np.zeros((h, d, q))
R = [[np.zeros((h))] * q] * d
D = np.zeros((d, q, 2))

# Convert L, R, D to matrix format
for d_idx, datum in enumerate(data):

    for h_idx, hypothesis in enumerate(gh.hypotheses):
        # L[h_idx, d_idx] = hypothesis.compute_likelihood(datum.data)
        L[d_idx][h_idx] = hypothesis.compute_likelihood(datum.data)

        for query, response, q_idx in datum.get_queries():
            # R[h_idx, d_idx, q_idx] = int(query in hypothesis())
            R[d_idx][q_idx][h_idx] = int(query in hypothesis())

    for query, response, q_idx in datum.get_queries():
        D[d_idx, q_idx, 0] = response[0]
        D[d_idx, q_idx, 1] = response[1]

stan_data = {
    'h': h,
    'r': r,
    'd': d,
    'q': q,
    'C': C,
    'L': L,
    'R': R
}

f2 = open('stan_data.p', 'w')
pickle.dump(stan_data, f2)

