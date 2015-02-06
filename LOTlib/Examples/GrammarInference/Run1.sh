#!/bin/bash

# alpha 0.9, domain 100
# LOT grammar, josh data
# sample 100k ngh via mcmc, collect top 1000 per chain, run 10 chains per data point, mcmc+mpi
python MakeNGHs.py -f out/ngh_lot100k.p -do 100 -a 0.9 -g lot_grammar -d josh_data -i 100000 -c 10 -n 1000 -mcmc -mpi

# LOT grammar, josh data
# sample 100k gh via mcmc, skip 100 cap 1000, pickle
python Run.py -q -p -csv out/gh_100k -ngh out/ngh_100k.p -g lot_grammar -d josh_data -i 100000 -sk 100 -cap 1000