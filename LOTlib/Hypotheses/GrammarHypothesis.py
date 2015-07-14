
"""
With this class, we can propose hypotheses as a vector of grammar rule probabilities.

Let's say that we have a set of 'domain hypotheses', for example representing a concept in the number
domain such as 'powers of two between 1 and 100'. These 'domain hypotheses' are found in `self.hypotheses`.

Each GrammarHypothesis stands for a model of hyperparameters - a 'parameter hypothesis' - used to generate a
set of domain hypotheses.


Methods
-------
__init__
propose
compute_prior
compute_likelihood
rule_distribution
set_value

get_rules

Note
----
This assumes:
    - domain hypothesis (e.g. NumberGameHypothesis) evaluates to a set
    - `data` is a list of FunctionData objects
    - input of FunctionData can be applied to hypothesis.compute_likelihood (for domain-level hypothesis)
    - output of FunctionData is a dictionary where each value is a pair (y, n),  where y is the number of
    positive and n is the number of negative responses for the given output key, conditioned on the input.

To use this for a different format, just change compute_likelihood.

"""
import copy
from math import exp, log
import random
import pickle
import numpy as np
from scipy.stats import gamma
from LOTlib.Hypotheses.VectorHypothesis import VectorHypothesis
from LOTlib.Miscellaneous import logsumexp, log1mexp, gammaln, Infinity, attrmem


class GrammarHypothesis(VectorHypothesis):
    """Hypothesis for grammar, we can represent grammar rule probability assignments as a vector.

    Inherits from VectorHypothesis, though I haven't figured out yet whe this really means...

    Attributes
    ----------
    grammar :  Grammar
        the grammar
    hypotheses : list<Hypothesis>
        list of hypotheses, generated beforehand
    rules : list<GrammarRule>
        list of all rules in the grammar
    value : np.ndarray
        vector of numbers corresponding to the items in `rules`
    proposal : np.ndarray
        proposal matrix of size:  |propose_idxs| x |propose_idxs|
    prior_shape : float
        shape used in compute_prior
    prior_scale : float
        shape used in compute_prior
    propose_n : int
    propose_scale : float


    """
    def __init__(self, grammar, hypotheses, rules=None, load=None, value=None, proposal=None,
                 prior_shape=2., prior_scale=1., propose_n=1, propose_scale=.1,
                 **kwargs):
        self.grammar = grammar
        if load:
            self.hypotheses = self.load_hypotheses(load)
        else:
            self.hypotheses = hypotheses
        self.rules = [r for sublist in grammar.rules.values() for r in sublist]
        if value is None:
            value = [rule.p for rule in self.rules]
        n = len(value)
        VectorHypothesis.__init__(self, value=value, n=n, proposal=proposal, propose_scale=propose_scale,
                                  propose_n=propose_n)
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.propose_n = propose_n
        self.propose_scale = propose_scale
        # self.compute_prior()
        self.update()

    # --------------------------------------------------------------------------------------------------------
    # MCMC-Related methods

    def __copy__(self, value=None):
        """Copy this GH; shallow copies of value & proposal so we don't have sampling issues."""
        if value is None:
            value = copy.copy(self.value)
        proposal = copy.copy(self.proposal)
        c = VectorHypothesis.__copy__(value)
        c.proposal = proposal
        return c

    def set_value(self, value):
        """Set value and grammar rules for this hypothesis."""
        assert len(value) == len(self.rules), "ERROR: Invalid value vector!!!"
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.rules = [r for sublist in self.grammar.rules.values() for r in sublist]
        self.update()

    def update(self):
        """Update `self.grammar` & priors for `self.hypotheses` relative to `self.value`.

        We'll need to do this whenever we calculate things like predictive, because we need to use
          the `posterior_score` of each domain hypothesis get weights for our predictions.

        """
        # Set probability for each rule corresponding to value index
        for i in range(1, self.n):
            self.rules[i].p = self.value[i]

        # Recompute prior for each hypothesis, given new grammar probs
        for h in self.hypotheses:
            h.compute_prior()
            h.update_posterior()

    # --------------------------------------------------------------------------------------------------------
    # Bayesian inference with GrammarHypothesis

    @attrmem('prior')
    def compute_prior(self):
        """
        Compute priors, according only to values that are proposed to. Priors computed according to gamma dist.

        """
        shape = self.prior_shape
        scale = self.prior_scale
        propose_idxs = [i for i, x in enumerate(self.get_propose_mask()) if x]

        propose_values = [self.value[i] for i in propose_idxs]
        rule_priors = [gamma.logpdf(v, shape, scale=scale) for v in propose_values]

        # If there are any negative values in our vector, prior is 0
        if [v for v in self.value if v < 0.0]:
            prior = -Infinity
        else:
            prior = sum([r for r in rule_priors])
        self.update_posterior()
        return prior

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """Use bayesian model averaging with `self.hypotheses` to estimate likelihood of generating the data.

        This is taken as a weighted sum over all hypotheses, sum_h { p(h | X) } .

        Args
        ----
        data : list
            List of FunctionData objects.
        update_post : bool
            Do we update `self.likelihood` and `self.posterior`?

        Returns
        -------
        float:
            Likelihood summed over all outputs, summed over all hypotheses & weighted for each
            hypothesis by posterior score p(h|X).

        Note
        ----
            This function is only designed to work with NumberGameHypothesis!

        """
        self.update()
        hypotheses = self.hypotheses
        likelihood = 0.0

        for d in data:
            posteriors = [sum(h.compute_posterior(d.data)) for h in hypotheses]
            Z = logsumexp(posteriors)
            weights = [(post - Z) for post in posteriors]

            for q, r in d.get_queries():
                # probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                p = logsumexp([w if q in h() else -Infinity for h, w in zip(hypotheses, weights)])
                p = -1e-10 if p >= 0 else p

                yes, no = r
                k = yes             # num. yes responses
                n = yes + no        # num. trials
                bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
                likelihood += bc + (k*p) + (n-k)*log1mexp(p)            # likelihood we got human output

        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    # --------------------------------------------------------------------------------------------------------
    # GrammarRule methods

    def get_rules(self, rule_name=False, rule_nt=False, rule_to=False):
        """Get all GrammarRules associated with this rule name, 'nt' type, and/or 'to' types.

        Note
        ----
        rule_name is a string, rule_nt is a string, rule_to is a string, though rule.to is a list.

        Returns
        ------
        idxs is a list of rules indexes, rules is a list of GrammarRules

        """
        rules = [(i, r) for i, r in enumerate(self.rules) if r.nt is not 'IGNORE']

        # Filter our rules that don't match our criteria
        if rule_name is not False:
            rules = [(i, r) for i, r in rules if r.name == rule_name]
        if rule_nt is not False:
            rules = [(i, r) for i, r in rules if r.nt == rule_nt]
        if rule_to is not False:
            if isinstance(rule_to, list):
                rules = [(i, r) for i, r in rules if all([rto in r.to for rto in rule_to])]
            else:
                rules = [(i, r) for i, r in rules if rule_to in r.to]

        # Zip rules into separate `idxs` & `rules` lists
        idxs, rules = zip(*rules) if len(rules) > 0 else [(), ()]
        return idxs, rules

    def get_propose_mask(self):
        """Only propose to rules with other rules with same NT."""
        propose_mask = [True] * self.n
        for i, nt in enumerate(self.grammar.nonterminals()):
            idxs, r = self.get_rules(rule_nt=nt)
            if len(idxs) == 1:
                propose_mask[i] = False
        return propose_mask

    def rule_distribution(self, data, rule_name, vals=np.arange(0, 2, .1)):
        """
        Compute posterior values for this grammar, varying specified rule over a set of values.

        Note
        ----
        This is not currently used...

        """
        idxs, rules = self.get_rules(rule_name=rule_name)
        assert len(idxs) == 1, "\nERROR: More than 1 GrammarRule with this name!\n"
        return self.conditional_distribution(data, idxs[0], vals=vals)

    # --------------------------------------------------------------------------------------------------------
    # Top domain hypotheses

    def get_top_hypotheses(self, n=10, key=(lambda x: x.posterior_score)):
        """Return the best `n` hypotheses from `self.hypotheses`."""
        self.update()
        hypotheses = self.hypotheses
        sorted_hypos = sorted(hypotheses, key=key)
        return sorted_hypos[-n:]

    def print_top_hypotheses(self, n=10):
        """Print the best `n` hypotheses from `self.hypotheses`."""
        self.update()
        for h in self.get_top_hypotheses(n=n):
            print str(h)
            print h.posterior_score, h.likelihood, h.prior

    def max_a_posteriori(self):
        self.update()
        return max([h.posterior_score for h in self.hypotheses])

    def max_like_estimate(self):
        self.update()
        return max([h.likelihood for h in self.hypotheses])

    # --------------------------------------------------------------------------------------------------------
    # Pickling domain hypotheses

    def load_hypotheses(self, filename=None):
        if filename:
            f = open(filename, "rb")
            self.hypotheses = pickle.load(f)
            for h in self.hypotheses:
                h.grammar = self.grammar
            return self.hypotheses

    def save_hypotheses(self, filename=None):
        if filename:
            f = open(filename, "wb")
            pickle.dump(self.hypotheses, f)

    # --------------------------------------------------------------------------------------------------------

    def csv_load(self, csv_file, map=True):
        """Load from CSV of values.

        Args
        ----
        csv_file : str
            where is the csv file we're loading values from?
        hypo_file : str
            where is the pickle file we're loading domain hypotheses from?



        Note
        ----
        - This assumes csv file is in the format: i,nt,name,to,p
        - There are

        """
        import csv

        # Get index
        max_idx = 1
        if map:
            with open(csv_file, 'rb') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['i'] > max_idx:
                        max_idx = row['i']
        print 'CSV row (max_idx): ', str(max_idx)

        with open(csv_file, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['i'] == max_idx:
                    i, r = self.get_rules(rule_name=row['name'], rule_nt=row['nt'], rule_to=eval(row['to']))
                    if len(r) == 1:
                        print 'Updating: ', r
                        print 'New p: ', row['p']
                        value = self.value
                        value[i[0]] = float(row['p'])
                        self.set_value(value)
                    else:
                        print 'ERROR in csv_load call to get_rules.'
                        print 'rule_name= ', row['name'], '\trule_nt= ', row['nt'], '\trule_to= ', row['to']
                        print 'rules: ', r

    def csv_compare_model_human(self, data, filename):
        """
        Save csv stuff for making the regression plot.

        Format is list of input/outputs, with human & model probabilities for each.

        Note
        ----
        This is specific to NumberGameHypothesis (because of 'o in h()')

        """
        import math
        import csv

        self.update()
        for h in self.hypotheses:
            h.compute_prior()
            h.update_posterior()

        with open(filename, 'a') as f:
            writer = csv.writer(f)
            hypotheses = self.hypotheses
            writer.writerow(['input', 'output', 'human p', 'model p'])
            i = 0

            for d in data:
                posteriors = [sum(h.compute_posterior(d.data)) for h in hypotheses]
                Z = logsumexp(posteriors)
                weights = [(post-Z) for post in posteriors]
                print i, '\t|\t', d.input
                i += 1

                for q, r in d.get_queries():
                    # Probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                    p_human = float(r[0]) / float(r[0] + r[1])
                    p_model = sum([math.exp(w) if q in h() else 0 for h, w in zip(hypotheses, weights)])
                    writer.writerow([d.data, q, p_human, p_model])




