"""
    Simple prolog example.

    I'm not sure why, but the use of a temporary file seems very finicky -- perhaps there is a problem
    with rapidly flushing the output, etc?
# """

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a simple grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

grammar = Grammar(start='PROGRAM')

grammar.add_rule('PROGRAM', '%s\n%s', ['LINE', 'PROGRAM'], 1.0)
grammar.add_rule('PROGRAM', '%s',       ['LINE'], 1.50)

grammar.add_rule('LINE',   '',  ['VARRULE'], 2.0)
grammar.add_rule('VARRULE','',  ['RULE'],    1.0, bv_type='ATOM', bv_prefix='X', bv_p=1.0) # a term with a variable

grammar.add_rule('RULE',   '',         ['VARRULE'], 0.50)
grammar.add_rule('RULE', '%s :- %s.',  ['HEAD', 'BODY'], 1.0)

grammar.add_rule('HEAD', '',  ['TERM'], 1.0)

grammar.add_rule('BODY', '',       ['TERM'], 1.0)
grammar.add_rule('BODY', '%s, %s', ['TERM', 'BODY'], 1.0)
grammar.add_rule('BODY', '%s; %s', ['TERM', 'BODY'], 1.0)


grammar.add_rule('TERM', '',  ['F/1'], 1.0)
grammar.add_rule('F/1', 'male', ['ATOM'], 1.0)
grammar.add_rule('F/1', 'female', ['ATOM'], 1.0)

grammar.add_rule('TERM', '',  ['F/2'], 1.0)
grammar.add_rule('F/2', 'grandparent', ['ATOM', 'ATOM'], 1.0)
grammar.add_rule('F/2', 'parent',      ['ATOM', 'ATOM'], 1.0)
# grammar.add_rule('F/2', 'sibling', ['ATOM', 'ATOM'], 1.0)
# grammar.add_rule('F/2', 'cousin', ['ATOM', 'ATOM'], 1.0)

PEOPLE = ['barak', 'michelle', 'sasha', 'malia', 'baraksr', 'ann', 'hussein', 'akumu']
for x in PEOPLE:
    grammar.add_rule('ATOM', x, None, 1.0/len(PEOPLE))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define PrologHypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from math import log
import pyswip

class PrologHypothesis(LOTHypothesis):

    def __init__(self, value=None, base_facts="", **kwargs):

        self.base_facts = base_facts # must be set before initializer

        LOTHypothesis.__init__(self, grammar, value=value, args=None, **kwargs)


    def __call__(self, expr):
        # Wrap expr in some limited inference and parse the output
        # TODO: Currently only handles a single QUERY as query variable

        # wrap to make it bounded: http://www.swi-prolog.org/pldoc/man?predicate=call_with_depth_limit/3
        thequery = "call_with_depth_limit(call_with_inference_limit(%s, 1000, _),1000,_)" % expr

        try:
            matches = list(self.fvalue.query(thequery))
        except pyswip.prolog.PrologError:
            return []
        # print matches

        # For mutiple solutions to "uncle(QUERY,john)", this will give me back something like
        # [{'QUERY': 'bob'}, {'QUERY': 'john'}, {'QUERY': 'mark'}]
        # so reformat
        return { a.get('QUERY') for a in matches if 'QUERY' in a }

    # def compile_function(self):
    #     ## Store the prolog interpreter as self.fvalue
    #     # tmpfile = "/tmp/lotlib-prolog-tmp33.pl" # "tmp"+re.sub(r"[\-0-9]", "", str(uuid.uuid1()))+".pl"
    #     tmpfile = "/tmp/lotlib-prolog-"+str(uuid.uuid1())+".pl"
    #     # This is probably slow, but we write it to a file
    #     with open(tmpfile, 'w') as f:
    #         f.write(self.base_facts)
    #         f.write("\n\n")
    #         f.write(str(self))
    #         f.write("\n\n")
    #         f.flush()
    #
    #     prolog = pyswip.Prolog()
    #     prolog.consult(tmpfile)
    #
    #     os.remove(tmpfile)
    #
    #     return prolog

    def compile_function(self):
        ## Store the prolog interpreter as self.fvalue

        # This is probably slow, but we write it to a file
        with open('tmp.pl', 'w') as f: ## TODO : replace with tempfile
            print >>f, self.base_facts
            print >>f, str(self), "\n"

        prolog = pyswip.Prolog()
        prolog.consult('tmp.pl')

        return prolog

    def compute_single_likelihood(self, datum):
        assert len(datum.input) == 1
        matches = self(*datum.input)

        p = (1.0-datum.alpha)*(1.0/len(PEOPLE)) # base rate
        if datum.output in matches: # or choose from the matches
            p += datum.alpha/len(matches)

        return log(p)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up default base facts and data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib import break_ctrlc


BASE_FACTS = """

:- discontiguous(female/1).
:- discontiguous(male/1).
:- discontiguous(parent/2).
:- discontiguous(grandparent/2).
:- style_check(-singleton).

spouse(barak, michelle).
male(barak).
female(michelle).
parent(michelle, sasha).
parent(michelle, malia).
parent(barak, sasha).
parent(barak, malia).
female(sasha).
female(malia).

parent(baraksr, barak).
parent(ann, barak).

parent(hussein, baraksr).
parent(akumu, baraksr).
"""

data = [FunctionData(input=["grandparent(baraksr, QUERY)"], output="sahsa", alpha=0.99),
        FunctionData(input=["grandparent(baraksr, QUERY)"], output="malia", alpha=0.99),
        FunctionData(input=["grandparent(ann, QUERY)"], output="sahsa", alpha=0.99),
        FunctionData(input=["grandparent(ann, QUERY)"], output="malia", alpha=0.99),
        FunctionData(input=["grandparent(hussein, QUERY)"], output="barak", alpha=0.99),
        FunctionData(input=["grandparent(akumu, QUERY)"], output="barak", alpha=0.99)
        ]


def make_hypothesis(**kwargs):
    return PrologHypothesis(base_facts=BASE_FACTS, **kwargs)

def make_data(n=1):
    return data*n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    h0 = make_hypothesis(likelihood_temperature=1.0)

    for h in break_ctrlc(MHSampler(h0, data)):
        print h
        print h.posterior_score, h.prior, h.likelihood, "\n"
        # h.value.fullprint()



