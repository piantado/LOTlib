from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Miscellaneous import Infinity
from optparse import OptionParser


parser = OptionParser()
parser.add_option("--steps", dest="STEPS", type="int", default=10000, help="Number of samples to run")
parser.add_option("--finite", dest="FINITE", type="int", default=20, help="specify the max_length to make language finite")
parser.add_option("--probe", dest="PROBE", type="int", default=50, help="after how many steps do we check the socre")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=20, help="Top number of hypotheses to store")
parser.add_option("--name", dest="NAME", type="string", default='', help="name of file")


def show_info(s):
    print '#'*90
    print '# ', s
    print '#'*90


def probe_MHsampler(h, get_data, options, size=64, data=None, init_size=None, iters_per_stage=None, sampler=None, ret_sampler=False):

    evaluation_data = get_data(size, max_length=options.FINITE)

    if data is None:
        if init_size is None:
            data = evaluation_data
        else:
            data = get_data(n=size, max_length=init_size)

    if sampler is None:
        sampler = MHSampler(h, data)

    best_hypotheses = TopN(N=options.TOP_COUNT)

    iter = 0
    rec = []

    for h in sampler:
        if iter == options.STEPS: break
        if iter % 100 == 0: print '---->', iter

        best_hypotheses.add(h)

        if iter % options.PROBE == 0:
            score_sum = 0
            for h in best_hypotheses:
                s = h.compute_posterior(evaluation_data)
                if s != -Infinity: score_sum += s
            rec.append([iter, score_sum])

        if init_size is not None and iter % iters_per_stage == 0:
            init_size += 2
            sampler.data = get_data(n=size, max_length=init_size)

        iter += 1

    if ret_sampler:
        return rec, sampler
    else:
        return rec