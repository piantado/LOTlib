
Grammar inference
===================

This contains developing code for performing grammar inference: given human responses, what should the PCFG prior probabilities be? 

This code has taken on various forms, including Hypotheses that implement grammar probabilities. 

However, as off Fall 2016, all code will use a CUDA implementation of grammar inference.

Data format
===========

The C code requires a hdf5 file with several files:

- specs - a list consisting of the number of hypotheses (NHyp), the number of grammar rules (Nrules), the number of responses (Ndata), the number of nonterminals (Nnt)
- counts - [NHyp x Nrules] - how often each nonterminal appears in each hypothesis - so that counts[h*Nrules+r] is the h'th hypothesis count of rule r
- output - [Ndata x NHyp] - the output of each hypothesis to each data point - so that output[Nhdata*h+d] is the h'th hypothesis' output to the d'th data point  (this ordering allows better CUDA memory access)
- human_yes - [Ndata] - number of "yes" responses by people
- human_no - [Ndata]  - number of "no" responses by people
- likelihood - [Ndata x NHyp] - the likelihood each hypothesis has on each data point - so that likelihood[Ndata*h+d] is the likelihood of the h'th hypothesis to the d'th data point
- ntlen - [Nrules] - how many rules of each type are there (in order by the columns of counts)

All arrays must be one-dimensional in C order. 
