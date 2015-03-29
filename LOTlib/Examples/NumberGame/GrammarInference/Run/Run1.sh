#!/bin/bash

# LOT grammar, josh data

# sample 100k ngh via mcmc, collect top 1000 per chain, run 10 chains per data point, mpi -n=16, alpha 0.9
mpiexec -n 16 python MakeNGHs.py -f out/ngh_lot100k.p -do 100 -a 0.9 -g lot_grammar -d josh_data -i 100000 -c 10 -n 1000 -mcmc -mpi

# sample 100k gh via mcmc, skip 100 cap 1000, pickle
python Run.py -q -p -csv out/gh_100k.p -ngh out/ngh_100k.p -g lot_grammar -d josh_data -i 100000 -sk 100 -cap 1000

# Generate .eps figure files for correlation plots (mix, indep, lot) & violin plots (indep, lot)
Rscript CSVtoFigures.R lot out/gh_lot100k