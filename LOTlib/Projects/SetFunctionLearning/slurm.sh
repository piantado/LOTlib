#!/bin/bash

#SBATCH -A colala --qos colala
#SBATCH -p colala

#SBATCH -J SetFunctionMakeHypotheses
#SBATCH -e hypotheses/slurm-err_%j
#SBATCH -o hypotheses/slurm-out_%j
#SBATCH --mem-per-cpu=1000
#SBATCH -t 500:15:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=24
#SBATCH --signal=2@600

module load numpy
module load python/2.7.10/b1
module load openmpi/1.6.5/b1

# This will run them sequentially
# srun time python MakeHypotheses.py --out=hypotheses/hypotheses-1.pkl   --steps=100000 --top=1 --chains=10
 srun time python MakeHypotheses.py --out=hypotheses/hypotheses-10.pkl  --steps=100000 --top=10 --chains=10
# srun time python MakeHypotheses.py --out=hypotheses/hypotheses-100.pkl --steps=100000 --top=100 --chains=10


exit
