
# A script to run all evaluation on a variety of models

MPIEXEC="mpiexec -n 16"
SAMPLES=100000
PRINT_EVERY=1000
REPITITIONS=25

for m in Number Galileo RationalRules RegularExpression SimpleMagnetism
do
	echo Running $m inference
	$MPIEXEC python inference-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPITITIONS
	R --no-save < plot-inference.R & 

	echo Running $m tempchain
	$MPIEXEC python tempchain-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPITITIONS
	R --no-save < plot-tempchain.R & 

	echo Running $m proposal
	$MPIEXEC python proposal-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPITITIONS
	R --no-save < plot-proposal.R & 
done
