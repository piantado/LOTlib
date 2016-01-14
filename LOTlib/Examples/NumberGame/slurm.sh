#!/bin/bash

#SBATCH -A colala --qos colala
#SBATCH -p colala

#SBATCH -J NumberGameMakeHypothesis
#SBATCH -e hypotheses/slurm-err_%j
#SBATCH -o hypotheses/slurm-out_%j
#SBATCH --mem-per-cpu=1000
#SBATCH -t 500:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --signal=2@600

module load numpy
module load python/2.7.10/b1
module load openmpi/1.6.5/b1

# This will run them sequentially
srun time python MakeHypotheses.py --grammar=lot_grammar --out=hypotheses/lot_hypotheses-10.pkl --steps=100000 --top=10 --chains=1
# srun time python MakeHypotheses.py --grammar=lot_grammar --out=hypotheses/lot_hypotheses-100.pkl --steps=100000 --top=100 --chains=1
# srun time python MakeHypotheses.py --grammar=lot_grammar --out=hypotheses/lot_hypotheses-1.pkl --steps=100000 --top=1 --chains=1
# 
# srun time python MakeHypotheses.py --grammar=mix_grammar --out=hypotheses/mix_grammar-10.pkl --steps=100000 --top=10 --chains=1
# 
# srun time python MakeHypotheses.py --grammar=independent_grammar --out=hypotheses/independent_grammar-10.pkl --top=10 --steps=100000 --chains=1


exit
