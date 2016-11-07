Grammar Inference
=================

This demo example lets you simulate people's generalizations and see the resulting grammar inference. 

This also provides documentation on the proper output format. The data.h5 format should contain:

    specs - array consisting of [<number of hypotheses>, <number of rules>, <number of human responses>, <number of nonterminals>]
    counts - counts of how often each rule is used in each hypothesis (computed with GrammarInference.Precompute)
    output - each hypothesis' output to each item people responded to [h1r1 h1r2 h1r3... ]
    human_yes - the number of times people said yes to the output i [o1 o2 o3 ..]
    human_no  - the number of times people said no to output i [o1 o2 o3 ...]
    likelihood - the likelihood each hypothesis had on each output response [h1r1 h1r2 h1r3 ...]
    ntlen - the number of rules with each nonterminal type (for re-normalizing pcfg)    

Run 
    python ExportToGPU.py 
in order to get data.h5.

Then run 
     ../main in=data.h5
to sample. 


Requirements
=================

cuda 8.0 (see NVIDIA's site)
LOTlib
h5py