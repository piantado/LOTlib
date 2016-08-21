from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.TopN import TopN
from LOTlib.Miscellaneous import logsumexp
from optparse import OptionParser
import numpy as np
from collections import Counter
from LOTlib.DataAndObjects import FunctionData
import LOTlib
from LOTlib.Inference.Samplers.StandardSample import standard_sample
from LOTlib.Projects.FormalLanguageTheory.Language.Index import instance

parser = OptionParser()
parser.add_option("--steps", dest="STEPS", type="int", default=10000, help="Number of samples to run")
parser.add_option("--finite", dest="FINITE", type="int", default=20, help="specify the max_length to make language finite")
parser.add_option("--probe", dest="PROBE", type="int", default=50, help="after how many steps do we check the socre")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=20, help="Top number of hypotheses to store")
parser.add_option("--name", dest="NAME", type="string", default='', help="name of file")
parser.add_option("--N", dest="N", type="int", default=3, help="number of inner hypotheses")
parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
CASE = 1


def show_info(s):
    print '#'*90
    print '# ', s
    print '#'*90


def to_file(rec, name):
    global CASE
    f = open(name, 'a')
    for e in rec:
        print >> f, CASE, e[0], e[1], e[2], e[3], e[4], e[5]
    f.close()

# TODO out of date
def probe_MHsampler(h, language, options, name, size=64, data=None, init_size=None, iters_per_stage=None, sampler=None, ret_sampler=False):
    get_data = language.sample_data_as_FuncData
    evaluation_data = get_data(size, max_length=options.FINITE)

    if data is None:
        if init_size is None:
            data = evaluation_data
        else:
            data = get_data(n=size, max_length=init_size)

    if sampler is None:
        sampler = MHSampler(h, data)
    else:
        sampler.data = data

    best_hypotheses = TopN(N=options.TOP_COUNT)

    iter = 0

    for h in sampler:
        if iter == options.STEPS: break
        if iter % 100 == 0: print '---->', iter

        best_hypotheses.add(h)

        if iter % options.PROBE == 0:

            for h in best_hypotheses:
                h.compute_posterior(evaluation_data)
            Z = logsumexp([h.posterior_score for h in best_hypotheses])

            pr_data = get_data(1024, max_length=options.FINITE)
            weighted_score = 0
            for h in best_hypotheses:
                precision, recall = language.estimate_precision_and_recall(h, pr_data)
                if precision + recall != 0:
                    f_score = precision * recall / (precision + recall)
                    weighted_score += np.exp(h.posterior_score - Z) * f_score
            weighted_score *= 2

            to_file([[iter, Z, weighted_score]], name)

        if init_size is not None and iter % iters_per_stage == 0:
            init_size += 2
            sampler.data = get_data(n=size, max_length=init_size)

        iter += 1

    if ret_sampler:
        return sampler


def probe(best_hypotheses, evaluation_data, pr_data, estimate_precision_and_recall):
    for h in best_hypotheses:
        h.compute_posterior(evaluation_data)
    Z = logsumexp([h.posterior_score for h in best_hypotheses])

    score_sum = 0
    best = 0
    s = None
    rec = []

    for h in best_hypotheses:
        precision, recall = estimate_precision_and_recall(h, pr_data)
        base = precision + recall

        if base != 0:
            p = np.exp(h.posterior_score - Z)
            weighted_score = p * (precision * recall / base)
            if weighted_score > best:
                best = weighted_score
                s = str(h)
            score_sum += weighted_score

            if p > 1e-2:
                rec.append([p, 2 * precision * recall / base])

    score_sum *= 2
    rec.sort(key=lambda x: x[0], reverse=True)
    return Z, score_sum, best*2, s, rec


def uniform_data(size, max_length=None):
    cnt = Counter()
    num = size * 2 / max_length
    for i in xrange(1, max_length/2+1):
        cnt['a'*i+'b'*i] = num
    return [FunctionData(input=[], output=cnt)]


def run(mk_hypothesis, size, finite, options, get_data=None, terminals=None):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(lambda: mk_hypothesis(options.LANG, N=options.N, terminals=terminals, bound=options.BOUND),
                           lambda: instance(options.LANG, finite).sample_data_as_FuncData(size) if get_data is None else get_data(size, max_length=options.FINITE),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=True, save_top=None, skip=200)
