
# A script to run all evaluation on a variety of models

MPIEXEC="time mpiexec -n 5" #"mpiexec -n 16"
SAMPLES=1000 
PRINT_EVERY=100 #1000
REPETITIONS=1 #25

for m in Number Galileo RationalRules RegularExpression SimpleMagnetism
do
	echo Running $m inference
	$MPIEXEC python inference-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
	R --no-save < plot-inference.R & 

	#echo Running $m tempchain
	#$MPIEXEC python tempchain-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
	#R --no-save < plot-tempchain.R & 

	#echo Running $m proposal
	#$MPIEXEC python proposal-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
	#R --no-save < plot-proposal.R & 
done
