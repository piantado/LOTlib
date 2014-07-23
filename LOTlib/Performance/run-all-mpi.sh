
# A script to run all evaluation on a variety of models

MPIEXEC="mpiexec -n 16"
SAMPLES=25000
REPITITIONS=10

for m in Number Galileo RationalRules SimpleMagnetism
do
	$MPIEXEC python inference-evaluation.py --model=$m --samples=$SAMPLES --print-every=1000 --repetitions=$REPITITIONS
	$MPIEXEC python tempchain-evaluation.py --model=$m --samples=$SAMPLES --print-every=1000 --repetitions=$REPITITIONS
done
