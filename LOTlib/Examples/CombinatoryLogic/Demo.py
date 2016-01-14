
from Grammar import *

if __name__ == "__main__":

    from CombinatoryLogic import combinator_reduce
    from LOTlib.Eval import EvaluationException

    for _ in range(10000):

        t = grammar.generate()

        lst = t.liststring()

        print lst, "\t->\t",
        try:
            print combinator_reduce(lst)
        except EvaluationException as e:
            print "*Probable-NON-HALT*"
