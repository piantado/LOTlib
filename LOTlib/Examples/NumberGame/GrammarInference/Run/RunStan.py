__author__ = 'Eric Bigelow'



# ============================================================================================================
# 1. Stan code
# ============

stan_code = """
data {
    int<lower=1> h;                     // # of domain hypotheses
    int<lower=1> r;                     // # of rules
    int<lower=0> d;                     // # of data points
    int<lower=0> q;                     // max # of queries for a given datum
    int<lower=0> C[h,r]                 // rule counts for each hypothesis
    real<upper=0> L[h,d];               // likelihood of data.input
    int<lower=0,upper=1> R[h,d,q];      // is each data.query in each hypothesis  (1/0)
    int<lower=0> D[d,q,2]               // human response for each data.query  (# yes, # no)
}


parameters {
    real<lower=0> x[r];                 // normalized vector of rule probabilities
}


model {

    // Prior
    increment_log_prob(gamma_log(lambda,alpha,beta));        // what are these args???

    // Likelihood model
    priors = C * x;                      // prior for each hypothesis

    for i in (1:d) {

        posteriors = L[:,i] + priors;
        Z = log_sum_exp(posteriors);
        w = exp(posteriors - Z);            // weights for each hypothesis

        for j in (1:q) {
            k = D[i,j,0];                   // num. yes responses
            n = D[i,j,0] + D[i,j,1];        // num. trials

            if ((n + k) > 0) {
                R_ij = w .* R[:, i, j];            // col `m` of boolean matrix `R[i]` weighted by `w`
                p = log(sum(R_ij));             // logsum of binary values for yes/no

                bc = tgamma(n+1) - (tgamma(k+1) + tgamma(n-k+1));       // binomial coefficient
                increment_log_prob(bc + (k*p) + (n-k)*log1m_exp(p));    // likelihood we got human output
            }
        }
    }

}
"""


# ============================================================================================================
# 2. Python code setting up variables and stuff
# =============================================

from LOTlib.DataAndObjects import HumanData
from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized
import numpy as np

data = [HumanData(1), HumanData(2), HumanData(3)]       # TODO
gh = GrammarHypothesisVectorized(args=[])               # TODO
gh.init_C()
all_queries = set.union( *[set(d.queries) for d in data] )

# int<lower=1> h;                     // # of domain hypotheses
# int<lower=1> r;                     // # of rules
# int<lower=0> d;                     // # of data points
# int<lower=0> q;                     // max # of queries for a given datum
# int<lower=0> C[h,r]                 // rule counts for each hypothesis
# real<upper=0> L[h,d];               // likelihood of data.input
# int<lower=0,upper=1> R[h,d,q];      // is each data.query in each hypothesis  (1/0)
# int<lower=0> D[d,q,2]               // human response for each data.query  (# yes, # no)

q = len(all_queries)
h = len(gh.hypotheses)
r = gh.n
d = len(data)
C = gh.C
L = np.zeros((r, d))
R = np.zeros((r, d, q))
D = np.zeros((d, q, 2))

# Convert L, R, D to matrix format
for d_idx, datum in enumerate(data):

    for h_idx, hypothesis in enumerate(gh.hypotheses):
        L[h_idx, d_idx] = hypothesis.compute_likelihood(datum.data)

        for query, response, q_idx in datum.get_queries():
            R[h_idx, d_idx, q_idx] = int(query in hypothesis())

    for query, response, q_idx in datum.get_queries():
        D[d_idx, q_idx, 0] = response[0]
        D[d_idx, q_idx, 1] = response[1]


# ============================================================================================================
# 3. Fitting the model in pystan!
# ===============================
import pystan

stan_data = {
    'h': h,
    'r': r,
    'd': d,
    'q': q,
    'C': C,
    'L': L,
    'R': R
}

fit = pystan.stan(model_code=stan_code, data=stan_data,
                  iter=1000, chains=4)

la = fit.extract(permuted=True)  # return a dictionary of arrays
mu = la['mu']

# return an array of three dimensions: iterations, chains, parameters
a = fit.extract(permuted=False)
print(fit)
fit.plot()















# ============================================================================================================
# X: pystan example code
# ======================
#
# schools_code = """
# data {
#     int<lower=0> J; // number of schools
#     real y[J]; // estimated treatment effects
#     real<lower=0> sigma[J]; // s.e. of effect estimates
# }
# parameters {
#     real mu;
#     real<lower=0> tau;
#     real eta[J];
# }
# transformed parameters {
#     real theta[J];
#     for (j in 1:J)
#     theta[j] <- mu + tau * eta[j];
# }
# model {
#     eta ~ normal(0, 1);
#     y ~ normal(theta, sigma);
# }
# """
#
# schools_dat = {'J': 8,
#                'y': [28,  8, -3,  7, -1,  1, 18, 12],
#                'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}
#
# fit = pystan.stan(model_code=schools_code, data=schools_dat,
#                   iter=1000, chains=4)

