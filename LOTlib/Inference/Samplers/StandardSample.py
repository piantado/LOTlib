import LOTlib
import pickle
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.TopN import TopN
from LOTlib.Miscellaneous import qq
from LOTlib import break_ctrlc
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
import operator  # for printing with getter in alsoprint


def standard_sample(make_hypothesis, make_data, show_skip=9, show=True, N=100, save_top='top.pkl', alsoprint='None', **kwargs):
    """
        Just a simplified interface for sampling, allowing printing (showing), returning the top, and saving.
        This is used by many examples, and is meant to easily allow running with a variety of parameters.
        NOTE: This skip is a skip *only* on printing
        **kwargs get passed to sampler
    """
    if LOTlib.SIG_INTERRUPTED:
        return TopN()  # So we don't waste time!

    h0 = make_hypothesis()
    data = make_data()


    best_hypotheses = TopN(N=N)

    f = eval(alsoprint)

    sampler = MHSampler(h0, data, **kwargs)

#    # TODO change acceptance temperature over times
#    sampler.acceptance_temperature = 0.5

    for i, h in enumerate(break_ctrlc(sampler)):

#        if i % 10000 == 0 and i != 0:
#            sampler.acceptance_temperature = min(1.0, sampler.acceptance_temperature+0.1)
#            print '='*50
#            print 'change acc temperature to', sampler.acceptance_temperature 

        best_hypotheses.add(h)

        if show and i%(show_skip+1) == 0:

            print i, \
                h.posterior_score, \
                h.prior, \
                h.likelihood, \
                f(h) if f is not None else '', \
                qq(cleanFunctionNodeString(h))

    if save_top is not None:
        print "# Saving top hypotheses"
        with open(save_top, 'w') as f:
            pickle.dump(best_hypotheses, f)

    return best_hypotheses
