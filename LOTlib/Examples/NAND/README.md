
Play around with learning multiple concepts from a single primitive like NAND. Useful for learning new chunks of concepts, a la Dechter, Malmaud, Adams & Tenenbaum (2013). LOTlib's scheme differs in that it computes the grammar adaptation that will minimize the KL-divergence between the average prior and the average posterior. 

Use Run.py to run inference on each concept in TargetConcepts and save the top hypotheses from each concept into all_hypotheses, which is written to pickle file hypotheses.pkl

You can then run Adapt.py, which reads hypotheses.pkl and calls OptimalGrammarAdaptation.print_subtree_adaptations to show the best subtrees to define for minimizing KL between the prior and posterior. Note that the output of Adapt.py should be sorted (e.g. via linux "sort"), with the lower number better)
