from pickle import load, dump
from collections import Counter

from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
import numpy as np
from LOTlib.Examples.FormalLanguageTheory.Language.SimpleEnglish import SimpleEnglish
register_primitive(flatten2str)
import matplotlib.pyplot as plt
from os import listdir
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
from LOTlib.Examples.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp
from LOTlib.Miscellaneous import Infinity
from math import log
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

suffix = time.strftime('_%m%d_%H%M%S', time.localtime())
fff = sys.stdout.flush


def load_hypo(_dir, keys):
    """
    1. read raw output file

    run: serial
    """
    rec = []

    for name in listdir(_dir):
        try:
            for e in keys:
                if e not in name: raise Exception
        except: continue

        print name; fff()
        rec.append([int(name.split('_')[1]), load(open(_dir+name))])

    return rec

# ===========================================================================
# staged & skewed
# ===========================================================================
def pos_seq(rec, infos):
    """
    1. evaluate posterior probability on different data sizes

    run: serial
    """
    seq = []
    seen = set()

    for e in rec:
        for h in e[1]:
            if h in seen: continue
            seen.add(h)

    for e in infos:
        prob_dict = {}
        language = AnBn(max_length=e[1])
        eval_data = language.sample_data_as_FuncData(e[0])

        for h in seen: prob_dict[h] = h.compute_posterior(eval_data)

        seq.append(prob_dict)
        print e, 'done'; fff()

    print '='*50
    return seq


def get_kl_seq():
    """
    1. read posterior sequences
    2. compute KL-divergence between adjacent distribution
    3. plot

    run: serial
    """
    seq_set = load(open('seq0_0825_164647')) + load(open('seq1_0825_164647')) + load(open('seq2_0825_164647'))
    kl_seq_set = []

    for seq in seq_set:
        kl_seq = []

        # add posterior set that without observing any data
        from copy import deepcopy
        dict_0 = deepcopy(seq[0])
        for h in dict_0:
            dict_0[h] = h.compute_posterior([FunctionData(input=[], output=Counter())])
        seq.insert(0, dict_0)

        # compute kl
        for i in xrange(len(seq)-1):
            current_dict = seq[i]
            next_dict = seq[i+1]
            current_Z = logsumexp([v for h, v in current_dict.iteritems()])
            next_Z = logsumexp([v for h, v in next_dict.iteritems()])

            kl = 0.0
            for h, v in current_dict.iteritems():
                p = np.exp(v - current_Z)
                if p == 0: continue
                kl += p * (v - next_dict[h] + next_Z - current_Z)

            kl_seq.append(log(kl))
            print 'KL from %i to %i: ' % (i, i+1), kl; fff()

        kl_seq_set.append(kl_seq)
        print '='*50; fff()

    staged, = plt.plot(range(12, 145, 2), kl_seq_set[0], label='staged')
    normal, = plt.plot(range(12, 145, 2), kl_seq_set[1], label='normal')
    uniform, = plt.plot(range(12, 145, 2), kl_seq_set[2], label='uniform')

    plt.legend(handles=[normal, staged, uniform])
    plt.ylabel('KL-divergence')
    plt.xlabel('data')
    plt.show()


def make_staged_seq():
    """
    1. read raw output
    2. evaluate posterior probability on different data sizes
    3. dump the sequence

    run: mpiexec -n 3
    """
    infos = [[i, 4*((i-1)/48+1)] for i in xrange(12, 145, 2)]
    work_list = [['staged'], ['normal0'], ['normal1']]
    rec = load_hypo('out/simulations/staged/', work_list[rank])
    seq = pos_seq(rec, infos)
    dump([seq], open('seq'+str(rank)+suffix, 'w'))


# ===========================================================================
# nonadjacent
# ===========================================================================
def make_pos():
    """
    1. read raw output
    2. compute precision & recall on nonadjacent and adjacent contents
    3. evaluate posterior probability on different data sizes
    4. dump the sequence

    run: serial
    """
    rec = load_hypo('out/simulations/nonadjacent/', ['0'])
    pr_dict = {}
    _set = set()
    cnt_tmp = {}
    for e in rec:
        for h in e[1]:
            if h in _set: continue
            cnt = Counter([h() for _ in xrange(1024)])
            cnt_tmp[h] = cnt
            base = sum(cnt.values())
            num = 0
            for k, v in cnt.iteritems():
                if k is None or len(k) < 2: continue
                if k[0] == 'a' and k[-1] == 'b': num += v
            pr_dict[h] = float(num) / base
            _set.add(h)

    space_seq = []
    for i in xrange(2, 65, 2):
        language = LongDependency(max_length=i)
        eval_data= language.sample_data_as_FuncData(144)
        prob_dict = {}
        ada_dict = {}

        for h in _set:
            prob_dict[h] = h.compute_posterior(eval_data)
            p, r = language.estimate_precision_and_recall(h, cnt_tmp[h])
            ada_dict[h] = p*r/(p+r) if p+r != 0 else 0

        space_seq.append([prob_dict, ada_dict])
        print i, 'done'; fff()

    dump(pr_dict, open('pr_dict', 'w'))
    dump(space_seq, open('space_seq', 'w'))


def dis_pos():
    """
    1. read posterior sequence
    2. set bound for axb and x hypotheses
    3. plot

    run: serial
    """
    space_seq = load(open('space_seq'))
    pr_dict = load(open('pr_dict'))

    seq = []
    seq1 = []
    seq2 = []
    for seen in space_seq:
        Z = logsumexp([p for h, p in seen[0].iteritems()])

        axb_prob = -Infinity
        x_prob = -Infinity
        for h, v in seen[0].iteritems():
            if pr_dict[h] > 0.2: axb_prob = logsumexp([axb_prob, v])
            if seen[1][h] > 0.2: x_prob = logsumexp([x_prob, v])

        seq.append(np.exp(axb_prob - Z))
        seq1.append(np.exp(x_prob - Z))
        seq2.append(np.exp(axb_prob - Z) - np.exp(x_prob - Z))
        print 'done'; fff()
    f, axarr = plt.subplots(1, 3)
    axarr[0].plot(range(2, 65, 1), seq)
    axarr[1].plot(range(2, 65, 1), seq1)
    axarr[2].plot(range(2, 65, 1), seq2)

    # plt.legend(handles=[x])
    plt.ylabel('posterior')
    plt.xlabel('poo_size')
    plt.show()


if __name__ == '__main__':
    # make_staged_seq()
    get_kl_seq()

    # make_pos()
    # dis_pos()
