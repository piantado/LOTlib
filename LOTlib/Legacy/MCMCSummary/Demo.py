if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_hypothesis, make_data
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

    from LOTlib.MCMCSummary import *

    for h in break_ctrlc(Print(PosteriorTrace(MHSampler(make_hypothesis(), make_data(100))))):
        pass

    # pt = PosteriorTrace(plot_every=1000, window=False)
    #
    # for h in break_ctrlc(pt(MHSampler(make_hypothesis(), make_data(100)))):
    #     print h
