from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Miscellaneous import Infinity, logsumexp
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


def to_file(rec, name):
    f = open(name, 'a')
    for e in rec:
        print >> f, e[0], e[1], e[2]
    f.close()


def estimate_precision_and_recall(h, data):
        """
        the precision and recall of h given a data set, it should be usually large
        """
        output = set(data[0].output.keys())
        h_out = set([h() for _ in xrange(int(sum(data[0].output.values())))])

        base = len(h_out)
        cnt = 0.0
        for v in h_out:
            if v in output: cnt += 1
        precision = cnt / base

        base = len(output)
        cnt = 0.0
        for v in output:
            if v in h_out: cnt += 1
        recall = cnt / base

        return precision, recall

def probe_MHsampler(h, get_data, options, name, size=64, data=None, init_size=None, iters_per_stage=None, sampler=None, ret_sampler=False):

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

    for h in sampler:
        if iter == options.STEPS: break
        if iter % 100 == 0: print '---->', iter

        best_hypotheses.add(h)

        if iter % options.PROBE == 0:
            for h in best_hypotheses:
                h.compute_posterior(evaluation_data)

            pr_data = get_data(1024, max_length=options.FINITE)
            score = 0
            for h in best_hypotheses:
                precision, recall = estimate_precision_and_recall(h, pr_data)
                score += precision + recall

            to_file([[iter, logsumexp([h.posterior_score for h in best_hypotheses]), score]], name)

        if init_size is not None and iter % iters_per_stage == 0:
            init_size += 2
            sampler.data = get_data(n=size, max_length=init_size)

        iter += 1

    if ret_sampler:
        return sampler