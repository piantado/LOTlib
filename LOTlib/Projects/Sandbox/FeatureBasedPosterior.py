"""
    Just a quick script to see how well a posterior can be modeled by a (sub)tree feature vector
"""

from LOTlib import break_ctrlc
import numpy
import pickle
from LOTlib.Miscellaneous import unique
from LOTlib.Examples.Number.Model import *
from LOTlib.Subtrees import *

# Make a set of "features" as partial subtrees
F = list(generate_unique_partial_subtrees(grammar, N=1000))

# Match the features on a tree
def get_feature_vector(t):
    return [count_subtree_matches(f, t) for f in F]

print "# Setting up hypothesis and data"
data = generate_data(10)

features = [] # (eventually) a matrix of features, one row per hypothesis
scores   = [] # (eventually) a vector of posterior scores

print "# Loading a pickle file"

with open('../Examples/Number/output/out.pkl', 'r') as f:
    hyps = pickle.load(f)
for h in hyps:
    h.compute_posterior(data)
    features.append( get_feature_vector(h.value) )
    scores.append(h.posterior_score)

# print "# Running MCMC # If you want to sample!
# h0 = NumberExpression(grammar)
# from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
# for h in break_ctrlc(unique(MHSampler(h0, data, steps=10000))): ## To re-run MCMC
#     features.append( get_feature_vector(h.value) )
#     scores.append(h.posterior_score)

print "# Converting variables"

X = numpy.matrix(features) # each hypothesis in a column
Y = numpy.array(scores)

# print X
# print Y

print "# Regressing"

from sklearn import linear_model

# Linear regression
# L = linear_model.LinearRegression(
L = linear_model.Ridge()
# L = linear_model.Lasso(alpha=0.01)

# fit it
L.fit(X,Y)

print "# -------- FIT COEFFICIENTS -------- "
# Print out the coefficients
for c, f in zip(L.coef_, F):
    print c, f
print "# R^2 = ", L.score(X,Y)