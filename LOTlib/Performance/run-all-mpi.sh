
# A script to run all evaluation on a variety of models

MPIEXEC="time nice -n 19 mpiexec -n 16" #"mpiexec -n 16"
SAMPLES=50000
PRINT_EVERY=1000 #1000
REPETITIONS=100 #25

for m in Number Galileo RationalRules RegularExpression SimpleMagnetism
do
	echo Running $m inference
	$MPIEXEC python inference-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
	R --no-save < plot-inference.R  

	echo Running $m tempchain
	$MPIEXEC python tempchain-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
	R --no-save < plot-tempchain.R  

	echo Running $m proposal
	# Don't run this on magnetism, since insert/delete doesn't work
	if [ $m -ne SimpleMagnetism]
	then
		$MPIEXEC python proposal-evaluation.py --model=$m --samples=$SAMPLES --print-every=$PRINT_EVERY --repetitions=$REPETITIONS
		R --no-save < plot-proposal.R  
	fi
done
