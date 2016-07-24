from pickle import load, dump
from collections import Counter

from LOTlib.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
import numpy as np
from LOTlib.Projects.FormalLanguageTheory.Language.SimpleEnglish import SimpleEnglish
register_primitive(flatten2str)
import matplotlib.pyplot as plt
from os import listdir
from LOTlib.Projects.FormalLanguageTheory.Language.AnBn import AnBn
from LOTlib.Projects.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp
from LOTlib.Miscellaneous import Infinity
from math import log
import sys
from mpi4py import MPI
from Simulation.utils import uniform_data
from optparse import OptionParser
from Language.Index import instance
import pstats, cProfile
import os
from copy import deepcopy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

suffix = time.strftime('_%m%d_%H%M%S', time.localtime())
fff = sys.stdout.flush

parser = OptionParser()
parser.add_option("--jump", dest="JUMP", type="int", default=2, help="")
parser.add_option("--temp", dest="TEMP", type="int", default=1, help="")
parser.add_option("--plot", dest="PLOT", type="string", default='yes', help="")
parser.add_option("--file", dest="FILE", type="string", default='', help="")
parser.add_option("--mode", dest="MODE", type="string", default='', help="")
parser.add_option("--axb", dest="AXB", type="float", default=0.3, help="")
parser.add_option("--x", dest="X", type="float", default=0.08, help="")
parser.add_option("--language", dest="LANG", type="string", default='An', help="")
parser.add_option("--finite", dest="FINITE", type="int", default=10, help="")
parser.add_option("--wfs", dest="WFS", type="string", default='yes', help="")
parser.add_option("--prob", dest="PROB", type="string", default='yes', help="")
parser.add_option("--stype", dest="STYPE", type="string", default='cube', help="")
parser.add_option("--dtype", dest="DTYPE", type="string", default='staged', help="")
(options, args) = parser.parse_args()

def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(int(slice_size)):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result

def load_hypo(_dir, keys):
    """
    1. read raw output file

    run: serial
    """
    rec = []

    for name in listdir(_dir):
        flag = False
        for e in keys:
            if e in name: flag = True; break
        if not flag: continue

        print name; fff()
        rec.append([int(name.split('_')[1]), load(open(_dir+name, 'r'))])

    return rec

# ===========================================================================
# prase SimpleEnglish
# ===========================================================================
def most_prob(suffix):
    topn = load(open('out/SimpleEnglish/hypotheses__' + suffix))
    Z = logsumexp([h.posterior_score for h in topn])
    h_set = [h for h in topn]; h_set.sort(key=lambda x: x.posterior_score, reverse=True)
    
    for i in xrange(10):
        print h_set[i]
        print 'prob`: ', np.exp(h_set[i].posterior_score - Z)
        print Counter([h_set[i]() for _ in xrange(512)])


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


def compute_kl(current_dict, next_dict):
    current_Z = logsumexp([v for h, v in current_dict.iteritems()])
    next_Z = logsumexp([v for h, v in next_dict.iteritems()])

    kl = 0.0
    for h, v in current_dict.iteritems():
        p = np.exp(v - current_Z)
        if p == 0: continue
        kl += p * (v - next_dict[h] + next_Z - current_Z)

    return kl


def get_kl_seq():
    """
    1. read posterior sequences
    2. compute KL-divergence between adjacent distribution
    3. plot

    run: serial
    """
    print 'loading..'; fff()
    seq_set = [load(open('seq0_0825_234606')), load(open('seq1_0825_234606')), load(open('seq2_0825_234606'))]
    # seq_set = load(open('seq0_0825_195540')) + load(open('seq1_0825_195540')) + load(open('seq2_0825_195540'))
    kl_seq_set = []

    print 'compute prior..'; fff()
    # add posterior set that without observing any data
    from copy import deepcopy
    dict_0 = deepcopy(seq_set[0][0])
    for h in dict_0:
        dict_0[h] = h.compute_posterior([FunctionData(input=[], output=Counter())])

    print 'making plot..'; fff()
    for seq in seq_set:
        kl_seq = []

        # # add posterior set that without observing any data
        # from copy import deepcopy
        # dict_0 = deepcopy(seq[0])
        # for h in dict_0:
        #     dict_0[h] = h.compute_posterior([FunctionData(input=[], output=Counter())])
        seq.insert(0, dict_0)

        # compute kl
        for i in xrange(len(seq)-1):
            current_dict = seq[i]
            next_dict = seq[i+1]
            kl = compute_kl(current_dict, next_dict)
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


def get_kl_seq2(jump, is_plot, file_name):
    print 'loading..'; fff()
    # TODO
    seq_set = [load(open('seq%i' % i + file_name)) for i in xrange(3)]
    kl_seq_set = []

    # print 'prior'
    # from copy import deepcopy
    # dict_0 = deepcopy(seq_set[0][0])
    # for h in dict_0:
    #     dict_0[h] = h.compute_posterior([FunctionData(input=[], output=Counter())])

    # avg_seq_set = []
    # for i in [4, 8, 12]:
    #     for j in xrange(i-4, i):
    #         sub = [seq_set[j] for j in xrange(i-4, i)]
    #         avg_seq_set.append([(a+b+c+d)/4 for a, b, c, d in zip(*sub)])

    for seq in seq_set:
        # seq.insert(0, dict_0)
        kl_seq = []

        kl_sum = 0

        for i in xrange(len(seq)-1):
            current_dict = seq[i]
            next_dict = seq[i+1]
            kl = compute_kl(current_dict, next_dict)
            kl_sum += kl; kl_seq.append(kl_sum)
            print 'KL from %i to %i: ' % (i, i+1), kl; fff()

            f = open('tmp_out', 'a')

            print >> f, kl_sum

        kl_seq_set.append(kl_seq)
        print '='*50; fff()
        f = open('tmp_out', 'a')
        print >> f, '='*50

    # print 'avging'; fff()
    # avg_seq_set = []
    # for i in [4, 8, 12]:
    #     sub = [kl_seq_set[j] for j in xrange(i-4, i)]
    #     avg_seq_set.append([(a+b+c+d)/4 for a, b, c, d in zip(*sub)])
    #
    # for avg_seq in avg_seq_set:
    #     for i in xrange(1, len(avg_seq)):
    #         avg_seq[i] = logsumexp([avg_seq[i], avg_seq[i-1]])
    #
    # dump(kl_seq_set, open('kl_seq_set'+suffix, 'w'))
    # dump(avg_seq_set, open('avg_seq_set'+suffix, 'w'))
    #
    if is_plot == 'yes':
        staged, = plt.plot([10**e for e in np.arange(0, 2.5, 0.1)], kl_seq_set[0], label='SI')
        normal, = plt.plot([10**e for e in np.arange(0, 2.5, 0.1)], kl_seq_set[1], label='SF')
        uniform, = plt.plot([10**e for e in np.arange(0, 2.5, 0.1)], kl_seq_set[2], label='RC')

        plt.legend(handles=[normal, staged, uniform])
        plt.ylabel('Cumulative KL-divergence')
        plt.xlabel('Size of Data')
        # plt.title('SS and SF')
        plt.show()


def make_staged_seq():
    """
    1. read raw output
    2. evaluate posterior probability on different data sizes
    3. dump the sequence

    run: mpiexec -n 3
    """
    infos = [[i, 4*((i-1)/48+1)] for i in xrange(12, 145, 2)]
    # infos = [[i, 12] for i in xrange(145, 278, 2)]
    work_list = [['staged'], ['normal0'], ['normal1']]
    rec = load_hypo('out/simulations/staged/', work_list[rank])
    seq = pos_seq(rec, infos)
    dump([seq], open('seq'+str(rank)+suffix, 'w'))


def make_staged_seq2(jump, temp):
    """
    run: mpiexec -n 12
    """
    rec = load_hypo('out/simulations/staged/', ['staged', 'normal0', 'normal1'])
    seen = set()
    work_list = slice_list(range(size), 3)

    for e in rec:
        for h in e[1]:
            if h in seen: continue
            seen.add(h)

    if rank in work_list[0]:
        seq = []
        infos = [[i, min(4*((int(i)-1)/48+1), 12)] for i in [10**e for e in np.arange(0, 2.2, 0.1)]]

        for e in infos:
            prob_dict = {}
            language = AnBn(max_length=e[1]+(e[1]%2!=0))
            eval_data = language.sample_data_as_FuncData(e[0])

            for h in seen:
                h.likelihood_temperature = temp
                prob_dict[h] = h.compute_posterior(eval_data)

            seq.append(prob_dict)
            print 'rank: ', rank, e, 'done'; fff()

    elif rank in work_list[1]:
        seq = []
        infos = [[i, 12] for i in [10**e for e in np.arange(0, 2.2, 0.1)]]

        for e in infos:
            prob_dict = {}
            language = AnBn(max_length=e[1])
            eval_data = language.sample_data_as_FuncData(e[0])

            for h in seen:
                h.likelihood_temperature = temp
                prob_dict[h] = h.compute_posterior(eval_data)

            seq.append(prob_dict)
            print 'rank: ', rank, e, 'done'; fff()

    else:
        seq = []
        infos = [[i, 12] for i in [10**e for e in np.arange(0, 2.2, 0.1)]]

        for e in infos:
            prob_dict = {}
            eval_data = uniform_data(e[0], e[1])

            for h in seen:
                h.likelihood_temperature = temp
                prob_dict[h] = h.compute_posterior(eval_data)

            seq.append(prob_dict)
            print 'rank: ', rank, e, 'done'; fff()

    # TODO no need ?
    from copy import deepcopy
    dict_0 = deepcopy(seq[0])
    for h in dict_0:
        dict_0[h] = h.compute_posterior([FunctionData(input=[], output=Counter())])
    seq.insert(0, dict_0)
    dump(seq, open('seq'+str(rank)+suffix, 'w'))


def test_sto():
    """
    objective: test if our posterior distribution is stable for each time of estimation

    run: mpiexec -n 12
    """
    rec = load_hypo('out/simulations/staged/', ['staged', 'normal0', 'normal1'])
    # rec = load_hypo('out/simulations/staged/', ['staged'])

    seen = set()
    for e in rec:
        for h in e[1]:
            if h in seen: continue
            seen.add(h)
    print rank, 'hypo len: ', len(seen); fff()

    seq = []
    inner_kl_seq = []
    infos = [[i, 4*((i-1)/48+1)] for i in xrange(12, 145, 2)]

    for e in infos:
        prob_dict = {}
        language = AnBn(max_length=e[1])
        eval_data = language.sample_data_as_FuncData(e[0])

        for h in seen:prob_dict[h] = h.compute_posterior(eval_data)

        seq.append(prob_dict)
        if len(seq) > 1: inner_kl_seq.append(compute_kl(seq[-2], seq[-1]))
        print 'rank: ', rank, e, 'done'; fff()

    dump(seq, open('seq_'+str(rank)+suffix,'w'))
    dump(inner_kl_seq, open('inner_kl_seq_'+str(rank)+suffix, 'w'))

    if rank != 0:
        comm.send(seq, dest=0)
        print rank, 'send'; fff()
        sys.exit(0)
    else:
        seq_set = [seq]
        for i_s in xrange(size - 1):
            seq_set.append(comm.recv(source=i_s+1))
            print rank, 'recv:', i_s; fff()

        cross_kl_seq = []
        for i_s in xrange(len(seq_set[0])):
            tmp = []

            for i_ss in xrange(len(seq_set)-1):
                current_dict = seq_set[i_ss][i_s]
                next_dict = seq[i_ss+1][i_s]
                tmp.append(compute_kl(current_dict, next_dict))
                print 'row %i column %i done' % (i_ss, i_s); fff()

            cross_kl_seq.append(tmp)

        dump(cross_kl_seq, open('cross_kl_seq_'+str(rank)+suffix, 'w'))
        for e in cross_kl_seq:
            print e; fff()


def test_hypo_stat():
    """
    objective: test how does those high prob hypotheses look like

    run: mpiexec -n 12
    """

    seq = load(open('seq_'+str(rank)+''))

    cnt = 0
    for e in seq:
        Z = logsumexp([p for h, p in e.iteritems()])
        e_list = [[h, p] for h, p in e.iteritems()]; e_list.sort(key=lambda x:x[1], reverse=True)
        f = open('hypo_stat_'+str(rank)+suffix, 'a')

        print >> f, '='*40
        for iii in xrange(4):
            print >> f, 'rank: %i' % rank, 'prob', np.exp(e_list[iii][1] - Z)
            print >> f, Counter([e_list[iii][0]() for _ in xrange(512)])
            print >> f, str(e_list[iii][0])

        print cnt, 'done'; cnt += 1
        f.close()


# ===========================================================================
# nonadjacent
# ===========================================================================
def make_pos(jump, temp):
    """
    1. read raw output
    2. compute precision & recall on nonadjacent and adjacent contents
    3. evaluate posterior probability on different data sizes
    4. dump the sequence

    run: mpiexec -n 4
    """

    print 'loading..'; fff()
    rec = load_hypo('out/simulations/nonadjacent/', ['0'])

    print 'estimating pr'; fff()
    pr_dict = {}
    _set = set()
    cnt_tmp = {}
    for e in rec:
        for h in e[1]:
            if h in _set: continue
            cnt = Counter([h() for _ in xrange(256)])
            # cnt = Counter([h() for _ in xrange(10)])
            cnt_tmp[h] = cnt
            base = sum(cnt.values())
            num = 0
            for k, v in cnt.iteritems():
                if k is None or len(k) < 2: continue
                if k[0] == 'a' and k[-1] == 'b': num += v
            pr_dict[h] = float(num) / base
            _set.add(h)

    work_list = range(2, 24, jump)
    space_seq = []
    for i in work_list:
        language = LongDependency(max_length=i)

        eval_data = {}
        for e in language.str_sets:
            eval_data[e] = 144.0 / len(language.str_sets)
        eval_data = [FunctionData(input=[], output=eval_data)]

        prob_dict = {}
        ada_dict = {}
        test_list = []

        for h in _set:
            h.likelihood_temperature = temp
            prob_dict[h] = h.compute_posterior(eval_data)
            p, r = language.estimate_precision_and_recall(h, cnt_tmp[h])
            ada_dict[h] = 2*p*r/(p+r) if p+r != 0 else 0

            test_list.append([h.posterior_score, ada_dict[h], pr_dict[h], cnt_tmp[h], str(h)])

        Z = logsumexp([h.posterior_score for h in _set])
        test_list.sort(key=lambda x:x[0], reverse=True)

        weighted_x = 0
        weighted_axb = 0
        for e in test_list:
            weighted_x += np.exp(e[0] - Z) * e[1]
            weighted_axb += np.exp(e[0] - Z) * e[2]
        f = open('non_w'+suffix, 'a')
        print >> f, weighted_x, weighted_axb
        f.close()
        # print rank, i, '='*50
        # for i_t in xrange(3):
        #     print 'prob: ', np.exp(test_list[i_t][0] - Z), 'x_f-score',  test_list[i_t][1], 'axb_f-score',  test_list[i_t][2]
        #     print test_list[i_t][3]
            # print test_list[i_t][5].compute_posterior(eval_data)
            # print language.estimate_precision_and_recall(test_list[i_t][5], cnt_tmp[test_list[i_t][5]])
        # fff()
        # dump(test_list, open('test_list_'+str(rank)+'_'+str(i)+suffix, 'w'))

        # space_seq.append([prob_dict, ada_dict])
        print 'rank', rank, i, 'done'; fff()

    dump([space_seq, pr_dict], open('non_seq'+str(rank)+suffix, 'w'))


def make_pos2(jump, temp):
    """
    1. read raw output
    2. compute precision & recall on nonadjacent and adjacent contents
    3. evaluate posterior probability on different data sizes
    4. dump the sequence

    run: mpiexec -n 4
    """

    print 'loading..'; fff()
    rec = load_hypo('out/simulations/nonadjacent/', ['0'])


    # TODO one do this
    print 'estimating pr'; fff()
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
                if k[0] + k[-1] in ['ab', 'cd', 'ef']: num += v
            pr_dict[h] = float(num) / base

            # fix the h_output
            h.h_output = cnt
            _set.add(h)

    work_list = range(2, 17, jump)
    for i in work_list:
        language = LongDependency(max_length=i)

        eval_data = {}
        for e in language.str_sets:
            eval_data[e] = 144.0 / len(language.str_sets)
        eval_data = [FunctionData(input=[], output=eval_data)]

        score = np.zeros(len(_set), dtype=np.float64)
        prec = np.zeros(len(_set), dtype=np.float64)

        # prob_dict = {}
        # test_list = []

        for ind, h in enumerate(_set):
            h.likelihood_temperature = temp
            score[ind] = h.compute_posterior(eval_data)
            prec[ind] = pr_dict[h]
            # prob_dict[h] = h.compute_posterior(eval_data)
            # test_list.append([h.posterior_score, pr_dict[h], cnt_tmp[h], str(h), h])

        # test_list.sort(key=lambda x: x[0], reverse=True)
        # Z = logsumexp([h.posterior_score for h in _set])
        #
        # weighted_axb = sum([np.exp(e[0] - Z) * e[1] for e in test_list])
        # print i, weighted_axb
        # for i_t in xrange(3):
        #     print 'prob: ', np.exp(test_list[i_t][0] - Z), 'axb_f-score',  test_list[i_t][1]
        #     print test_list[i_t][2]
        #     # print test_list[i_t][4].compute_posterior(eval_data)
        #     # print language.estimate_precision_and_recall(test_list[i_t][5], cnt_tmp[test_list[i_t][5]])
        # print '='*50
        # fff()

        #
        # f = open('non_w'+suffix, 'a')
        # print >> f, Z, weighted_axb
        # print
        # f.close()
        #
        # print 'size: %i' % i, Z, weighted_axb; fff()

        if rank != 0:
            comm.send(score, dest=0)
            comm.send(prec, dest=0)
            sys.exit(0)
        else:
            for r in xrange(size - 1):
                score += comm.recv(source=r+1)
                prec += comm.recv(source=r+1)
            score /= size
            prec /= size
            Z = logsumexp(score)

            weighted_axb = np.sum(np.exp(score-Z) * prec)

            f = open('non_w'+suffix, 'a')
            print >> f, Z, weighted_axb
            print i, Z, weighted_axb; fff()
            f.close()
        # dump([space_seq, pr_dict], open('non_seq'+str(rank)+suffix, 'w'))


def parse_nonadjacent(temperature):
    """
        load the hypothesis space and compute weighted F-scores of nonadjacent dependency on different pool sizes.
        replace the make_pos function

        example script:
            mpiexec -n 12 python parse_hypothesis.py --mode=nonadjacent_mk --temp=100
    """
    eval_data_size = 1024
    global size
    global rank
    pr_dict = {}
    _set = set()

    if rank == 0:
        print 'loading..'; fff()
        rec = load_hypo('out/simulations/nonadjacent/', ['_'])

        print 'estimating pr'; fff()

        for e in rec:
            for h in e[1]:

                if h in _set: continue

                cnt = Counter([h() for _ in xrange(eval_data_size)])
                num = 0
                for k, v in cnt.iteritems():
                    if k is None or len(k) < 2: continue
                    if k[0] + k[-1] in ['ab', 'cd', 'ef']: num += v

                pr_dict[h] = float(num) / eval_data_size
                _set.add(h)

        #debug
        _list = [[h, pr] for h, pr in pr_dict.iteritems()]; _list.sort(key=lambda x: x[1], reverse=True)
        for i in xrange(10):
            print 'p,r: ', _list[i][1],
            print Counter([_list[i][0]() for _ in xrange(256)])
            print _list[i][0]
        print '='*50; fff()

    print "sync..."; fff()
    pr_dict = comm.bcast(pr_dict, root=0)
    _set = comm.bcast(_set, root=0)

    # work_list = slice_list(np.arange(2, 65, 2), size)
    work_list = slice_list(np.arange(10, 66, 5), size)
    seq = []
    for s in work_list[rank]:
        wfs = 0.0
        language = LongDependency(max_length=s)
        eval_data = [FunctionData(input=[], output={e: float(eval_data_size)/s for e in language.str_sets})]

        for h in _set:
            h.likelihood_temperature = temperature
            h.compute_posterior(eval_data)

        Z = logsumexp([h.posterior_score for h in _set])
        seq.append([s, sum([pr_dict[h]*np.exp(h.posterior_score - Z) for h in _set])])

        #debug
        _list = [h for h in _set]; _list.sort(key=lambda x: x.posterior_score, reverse=True)
        print 'pool size: ', s
        for i in xrange(3):
            print 'prob: ', np.exp(_list[i].posterior_score - Z), 'p,r: ', pr_dict[_list[i]],
            print Counter([_list[i]() for _ in xrange(256)])
            print _list[i]
        print '='*50; fff()

    if rank == 0:
        for i in xrange(1, size):
            seq += comm.recv(source=i)
    else:
        comm.send(seq, dest=0)
        sys.exit(0)

    seq.sort(key=lambda x: x[0])
    f = open('nonadjacent_wfs_seq'+suffix, 'w')
    for s, wfs in seq:
        print >> f, s, wfs
    f.close()


def dis_pos(jump, is_plot, file_name, axb_bound, x_bound):
    """
    1. read posterior sequence
    2. set bound for axb and x hypotheses
    3. plot

    run: serial
    """
    # space_seq, pr_dict = load(open('non_seq%i_' % i + file_name))
    print 'loading..'; fff()
    _set = [load(open('non_seq%i_' % i + file_name)) for i in xrange(4)]

    print 'avging..'; fff()
    avg_space_seq, avg_pr_dict = _set.pop(0)
    for space_seq, pr_dict in _set:
        for i in xrange(len(space_seq)):
            prob_dict, ada_dict = space_seq[i]
            avg_prob_dict, avg_ada_dict = avg_space_seq[i]
            for h in prob_dict:
                avg_prob_dict[h] = logsumexp([avg_prob_dict[h], prob_dict[h]])
                avg_ada_dict[h] += ada_dict[h]
        for h in pr_dict:
            avg_pr_dict[h] += pr_dict[h]

    for prob_dict, ada_dict in avg_space_seq:
        for h in prob_dict:
            prob_dict[h] -= 4
            ada_dict[h] /= 4
    for h in avg_pr_dict:
        avg_pr_dict[h] /= 4

    for axb_bound in np.arange(0.1, 1, 0.1):
        for x_bound in np.arange(0.1, 1, 0.1):
            seq = []
            seq1 = []
            seq2 = []
            for seen in avg_space_seq:
                Z = logsumexp([p for h, p in seen[0].iteritems()])

                axb_prob = -Infinity
                x_prob = -Infinity

                for h, v in seen[0].iteritems():
                    if avg_pr_dict[h] > axb_bound: axb_prob = logsumexp([axb_prob, v])
                    if seen[1][h] > x_bound: x_prob = logsumexp([x_prob, v])

                seq.append(np.exp(axb_prob - Z))
                seq1.append(np.exp(x_prob - Z))
                seq2.append(np.exp(axb_prob - Z) - np.exp(x_prob - Z))
                print 'done'; fff()

            flag = True
            for i in xrange(len(seq2) - 1):
                if seq2[i] - seq2[i+1] > 1e-4:
                    flag = False; break
            if not flag: continue

            print axb_bound, x_bound, '='*50
            print 'axb_prob: ', seq
            print 'x_prob: ', seq1
            print 'difference_prob: ', seq2; fff()
            dump([seq, seq1, seq2], open('nonadjacent_%.2f_%.2f' % (axb_bound, x_bound)+suffix, 'w'))

            if is_plot == 'yes':
                f, axarr = plt.subplots(1, 3)
                axarr[0].plot(range(2, 65, jump), seq)
                axarr[1].plot(range(2, 65, jump), seq1)
                axarr[2].plot(range(2, 65, jump), seq2)

                # plt.legend(handles=[x])
                plt.ylabel('posterior')
                plt.xlabel('poo_size')
                plt.show()


def test_lis_disp(names):
    ll = [load(open(name)) for name in names]
    for li in ll:
        print '='*50
        Z = logsumexp([h[0] for h in li])
        for i in xrange(3):
                print 'p ', np.exp(li[i][0] -Z), 'x_f-score ', li[i][1], 'axb_f-score', li[i][2]
                print li[i][4]

# ===========================================================================
# parse & plot
# ===========================================================================
def parse_plot(lang_name, finite, is_plot):
    """
        run: mpi supported

        example:
            mpiexec -n 12 python parse_hypothesis.py --mode=parse_plot --language=An --finite=3 --plot=yes --wfs=yes
    """
    _dir = 'out/final/'
    global size
    global rank
    topn = set()
    prf_dict = {}
    language = instance(lang_name, finite)

    if rank == 0:
        print 'loading..'; fff()
        for file_name in listdir(_dir):
            if lang_name + '_' in file_name:
                _set = load(open(_dir+file_name))
                topn.update([h for h in _set])

        print 'getting p&r..'; fff()
        pr_data = language.sample_data_as_FuncData(1024)
        for h in topn:
            p, r = language.estimate_precision_and_recall(h, pr_data)
            prf_dict[h] = [p, r, 0 if p+r == 0 else 2*p*r/(p+r)]

        dump(prf_dict, open(lang_name+'_prf_dict'+suffix, 'w'))

    topn = comm.bcast(topn, root=0)
    prf_dict = comm.bcast(prf_dict, root=0)

    print rank, 'getting posterior'; fff()
    work_list = slice_list(np.arange(235, 300, 5), size)
    seq = []

    pnt_str = 'Weighted F-score' if options.WFS == 'yes' else 'Posterior Probability'
    for s in work_list[rank]:
        eval_data = language.sample_data_as_FuncData(s)
        for h in topn:
            h.likelihood_temperature = 100
            h.compute_posterior(eval_data)

        Z = logsumexp([h.posterior_score for h in topn])

        if options.WFS == 'yes': tmp = sum([prf_dict[h][2]*np.exp(h.posterior_score - Z) for h in topn])
        # TODO
        # else: tmp = sum([np.exp(h.posterior_score - Z) for h in topn if prf_dict[h][2] > 0.9])
        else: tmp = sum([np.exp(h.posterior_score - Z) for h in topn if (prf_dict[h][0] < 0.3 and prf_dict[h][1] > 0.9)])

        if options.PROB == 'yes': dump([topn, Z], open(lang_name+'_prob_'+str(s)+suffix, 'w'))

        seq.append([s, tmp])
        print 'size: %.1f' % s, '%s: %.2f' % (pnt_str, tmp); fff()

        #debug
        _list = [h for h in topn]; _list.sort(key=lambda x: x.posterior_score, reverse=True)
        for i in xrange(3):
            print 'prob: ', np.exp(_list[i].posterior_score - Z), 'p,r: ', prf_dict[_list[i]][:2],
            print Counter([_list[i]() for _ in xrange(256)])
            print _list[i]
        print '='*50; fff()

    if rank == 0:
        for i in xrange(1, size):
            seq += comm.recv(source=i)
    else:
        comm.send(seq, dest=0)
        sys.exit(0)

    seq.sort(key=lambda x: x[0])
    dump(seq, open(lang_name+'_seq'+suffix, 'w'))

    if is_plot == 'yes':
        x, y = zip(*seq)
        plt.plot(x, y)

        plt.ylabel(pnt_str)
        plt.xlabel('Size of Data')
        plt.title(lang_name)
        plt.show()


def parse_cube(lang_name, finite):
    """
        reads hypotheses of lang_name, estimates the p/r and posterior score, and saves them into a cube (list of tables)

        data structure:
            stats = [cube, topn]
            cube = [[size, Z, table], [size, Z, table], ...]
            table = [[ind, score, p, r, f, strs], [ind, score, p, r, f, strs], ...]

        NOTE: topn here is a dict, you can use ind to find the h

        example script:
            mpiexec -n 12 python parse_hypothesis.py --mode=parse_cube --language=An --finite=3/10
    """
    _dir = 'out/'
    global size
    global rank
    topn = dict()
    prf_dict = {}
    language = instance(lang_name, finite)

    if rank == 0:
        truncate_flag = False if (lang_name == 'An' and finite <= 3) else True
        set_topn = set()
        print 'loading..'; fff()
        for file_name in listdir(_dir):
            if lang_name + '_' in file_name:
                _set = load(open(_dir+file_name))
                set_topn.update([h for h in _set])

        print 'getting p&r..'; fff()
        pr_data = language.sample_data_as_FuncData(2048)
        for h in set_topn:
            p, r, h_llcounts = language.estimate_precision_and_recall(h, pr_data, truncate=truncate_flag)
            prf_dict[h] = [p, r, 0 if p+r == 0 else 2*p*r/(p+r)]
            h.fixed_ll_counts = h_llcounts

        topn = dict(enumerate(set_topn))
        print 'bcasting..'; fff()

    topn = comm.bcast(topn, root=0)
    prf_dict = comm.bcast(prf_dict, root=0)

    print rank, 'getting posterior'; fff()
    # work_list = slice_list(np.arange(0, 72, 6), size)
    work_list = slice_list(np.arange(120, 264, 12), size)

    cube = []
    for s in work_list[rank]:
        eval_data = language.sample_data_as_FuncData(s)
        for ind, h in topn.iteritems():
            h.likelihood_temperature = 100
            h.compute_posterior(eval_data)

        Z = logsumexp([h.posterior_score for ind, h in topn.iteritems()])

        table = [[ind, h.posterior_score, prf_dict[h][0], prf_dict[h][1], prf_dict[h][2], h.fixed_ll_counts] for ind, h in topn.iteritems()]
        table.sort(key=lambda x: x[1], reverse=True)
        cube += [[s, Z, table]]

        print rank, s, 'done'; fff()

    if rank == 0:
        for i in xrange(1, size):
            cube += comm.recv(source=i)
    else:
        comm.send(cube, dest=0)
        print rank, 'table sent'; fff()
        sys.exit(0)

    cube.sort(key=lambda x: x[0])
    dump([cube, topn], open(lang_name+'_stats'+suffix, 'w'))


def cube2graph(file_name, plot):
    """
        read the stats file and make graph & table

        example script:
            python parse_hypothesis.py --mode=cube2graph --file=file --plot=no
    """
    cube, topn = load(open('./stats/'+file_name))
    # cube, topn = load(open('./'+file_name))

    x = []
    wfs = []
    wp = []
    wr = []
    top10_f = [[] for _ in xrange(10)]
    for size, Z, table in cube:
        x.append(size)
        # # for anbncn
        # for i in xrange(len(table)):
        #     if  -0.02 < table[i][4] - 0.92 < 0.02: table[i][4]=table[i][2]=table[i][3]=1.0
        wfs.append(sum([row[4]*np.exp(row[1] - Z) for row in table]))
        wp.append(sum([row[2]*np.exp(row[1] - Z) for row in table]))
        wr.append(sum([row[3]*np.exp(row[1] - Z) for row in table]))
        for i in xrange(10): top10_f[i].append(table[i][4])

    MAP_f = top10_f.pop(0)
    # fig_other_f = None
    # for e in top10_f:
    #     fig_other_f, = plt.plot(x, e, '#B0B0B0', label='other hypothesis F-score')

    # smooth MAP_f
    # flag = False
    # for i in xrange(len(MAP_f)):
    #     if -1e-2 < MAP_f[i] - 0.93 < 1e-2: flag = True
    #     if flag: MAP_f[i] = 1.0

    # make latex table
    f_table = open(file_name+'_toph_table', 'w')
    _, __, table = cube[-1]
    for ind, score, p, r, f, strs in table:
        print >> f_table, str(topn[ind]), '&', '%.2f' % score, '&', '%.2f' % p, '&', '%.2f' % r, '&', '%.2f' % f, '&', '%.2f' % np.exp(score - __), '&', '$(',

        strs = [v for v in strs.keys()]; strs.sort(key=lambda x:len(x))
        strs = ['\\mb{'+v+'}' for v in strs]
        if strs[0] == '\\mb{}': strs[0] = '\\emptyset'
        l = strs.pop()
        for s in strs: print >> f_table, s, ',',
        print >> f_table, l, ')$', '\\\\'
    f_table.close()


    dump([x, wfs, wp, wr, MAP_f], open(file_name+'_graph', 'w'))
    if plot != 'yes': sys.exit(0)

    fig_wfs, = plt.plot(x, wfs, 'r', label='Weighted F-score')
    fig_wp, = plt.plot(x, wp, 'r--', label='Weighted precision')
    fig_wr, = plt.plot(x, wr, 'r-.', label='Weighted recall')
    fig_map_f, = plt.plot(x, MAP_f, label='MAP hypothesis F-score')

    # f = open('mod','w')
    # for i in xrange(len(wfs)):
    #     print >> f, x[i], wfs[i], wp[i], wr[i], MAP_f[i]
    # f.close()

    plt.ylabel('Scores')
    plt.xlabel('Amount of data')
    plt.legend(handles=[fig_wfs, fig_wp, fig_wr, fig_map_f], loc=5)
    # plt.axis([0, 30, 0, 1.05])
    plt.axis([0, 240, 0, 1.05])
    matplotlib.rcParams.update({'font.size': 16})



    plt.show()




def only_plot(file_name, stype):
    """
        plot graph based on stats

        python parse_hypothesis.py --mode=only_plot --file=file --stype=cube
    """

    if stype == 'cube':
        x, wfs, wp, wr, MAP_f = load(open(file_name))
        fig_wfs, = plt.plot(x, wfs, 'r', label='Weighted F-score')
        fig_wp, = plt.plot(x, wp, 'r--', label='Weighted precision')
        fig_wr, = plt.plot(x, wr, 'r-.', label='Weighted recall')
        fig_map_f, = plt.plot(x, MAP_f, label='MAP hypothesis F-score')

        plt.ylabel('Scores')
        plt.xlabel('Size of Data')
        plt.legend(handles=[fig_wfs, fig_wp, fig_wr, fig_map_f], loc=5)
        plt.axis([0, 66, 0, 1.05])
        # plt.axis([0, 240, 0, 1.05])
        matplotlib.rcParams.update({'font.size': 16})
        plt.show()

    # x = []; y = []
    #
    # if is_txt == 'yes':
    #     f = open(lang_name+suffix)
    #     for line in f:
    #         a, b = line.split()
    #         x.append(float(a))
    #         y.append(float(b))
    # else:
    #     seq = load(open(lang_name+'_seq'+suffix))
    #     x, y = zip(*seq)
    #
    # plt.plot(x, y)
    #
    # plt.ylabel('Weighted F-score')
    # # plt.ylabel('Posterior Probability')
    # plt.xlabel('Size of Data')
    # plt.title(lang_name)
    # plt.show()


def temp():
    seq = []
    for s in range(0, 61, 2):
        print 'loading %i' % s
        seq.append(load(open('An_prob_%i_1106_143007' % s)))
    _dict = load(open('An_prf_dict_1106_143007'))
    return seq, _dict

    # _dir = 'out/final/'
    # lang_name = 'Dyck'
    #
    # print 'loading..'; fff()
    # topn = set()
    # for file_name in listdir(_dir):
    #     if lang_name in file_name:
    #         _set = load(open(_dir+file_name))
    #         topn.update([h for h in _set])
    #
    # print 'prob..'
    # language = instance(lang_name, 8)
    # eval_data = language.sample_data_as_FuncData(20)
    # for h in topn:
    #     h.likelihood_temperature = 100
    #     h.compute_posterior(eval_data)
    #
    # Z = logsumexp([h.posterior_score for h in topn])
    # #debug
    # _list = [h for h in topn]; _list.sort(key=lambda x: x.posterior_score, reverse=True)
    # for i in xrange(20):
    #     print 'prob: ', np.exp(_list[i].posterior_score - Z)
    #     print Counter([_list[i]() for _ in xrange(256)])
    #     print _list[i]


def inf_prob(seq, _dict, flag, p1, p2):
    ind = 0
    for topn, Z in seq:
        if flag: tmp = sum([np.exp(h.posterior_score - Z) for h in topn if _dict[h][2] > p1])
        else: tmp = sum([np.exp(h.posterior_score - Z) for h in topn if (_dict[h][0] < p1 and _dict[h][1] > p2)])
        print ind, ': ', tmp; ind += 2


# ==============================================
# parse staged input case, a version made in 2016/04/26
# ==============================================

def make_staged_posterior_seq(_dir, temperature, lang_name, dtype):
    """
        script: python parse_hypothesis.py --mode=make_staged_posterior_seq --file=file --temp=1 --language=AnBn --dtype=staged/uniform

        1. read raw file
        2. compute fixed Counter
        3. compute posterior for different amounts

        dumped posterior format: [topn, [z,amount,finite,[s1,s2,....]], [], [], ....]

        NOTE: if _dir is previously dumped posterior seq, then we use it
    """
    
    if not (os.path.isfile(_dir) and 'posterior_seq' in _dir):

        topn = set()

        for filename in os.listdir(_dir):
            if ('staged' in filename or 'normal' in filename) and 'seq' not in filename:
                print 'load', filename
                _set = load(open(_dir+filename))
                topn.update([h for h in _set])
        topn = list(topn)

        # fix the llcnts to save time and make curve smooth
        print 'get llcnts...'
        for h in topn:
            llcnts = Counter([h() for _ in xrange(2048)])
            h.fixed_ll_counts = llcnts
            
            
        seq = []
        seq.append(topn)

        for amount, finite in mk_staged_wlist(0,200,2,[48,96]):
            
            print 'posterior on', amount, finite
            
            if dtype == 'staged':
                language = instance(lang_name, finite)
                eval_data = language.sample_data_as_FuncData(amount)
            elif dtype == 'uniform':
                eval_data = uniform_data(amount, 12)

            for h in topn:
                h.likelihood_temperature = temperature
                h.compute_posterior(eval_data)

            Z = logsumexp([h.posterior_score for h in topn])
            seq.append([Z, amount, finite, [h.posterior_score for h in topn]])

        dump(seq,open(dtype + '_posterior_seq' + suffix, 'w'))

    else:
        seq = load(open(_dir))
    

    # ====================== compute KL based on seq =======================
    
    print 'compute kl seq...'
    kl_seq = []
    topn = seq.pop(0)
    for i in xrange(len(seq)-1):
        kl_seq.append([seq[i][1],compute_kl2(seq[i],seq[i+1])])

    dump(kl_seq,open(dtype + '_kl_seq' + suffix,'w'))




def compute_kl2(now, nxt):
    now_z, _, __, now_scores = now
    nxt_z, _, __, nxt_scores = nxt

    kl = 0.0

    for i in xrange(len(now_scores)):
        p = np.exp(now_scores[i] - now_z)
        if p == 0: continue
        kl += p * (now_scores[i] - nxt_scores[i] + nxt_z - now_z)

    return kl

    



def mk_staged_wlist(left, right, step, stage):
    stage1, stage2 = stage
    w_list = [[i, 0] for i in xrange(left, right, step)]
    
    ind = 0
    
    for amount, _ in deepcopy(w_list):
        if amount < stage1:
            w_list[ind][1] = 4
        elif amount < stage2:
            w_list[ind][1] = 8
        else:
            w_list[ind][1] = 12
        ind += 1

    return w_list


def plt_kl_seq(file1, file2):
    
    

    staged_seq = load(open(file1))
    uniform_seq = load(open(file2))

    x, y = zip(*staged_seq)
    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(x, y)
    ax1.set_title('staged')

    x, y = zip(*uniform_seq)
    ax2.plot(x, y)
    ax2.set_title('uiform')

    plt.show()
    

# ======================================================
# parse nonadjacent case
# ======================================================

def parse_nonadjacent(_dir,temperature):
    """
        1. read raw hypos
        2. get fixed llcnts
        3. compute posterior given different data pool sizes

        NOTE: if _dir is previously dumped topn then load it
    """

    if 'nonadjacent_topn' not in _dir:
        topn = set()
        for filename in os.listdir(_dir):
            if 'nonadjacent' in filename and 'seq' not in filename:
                print 'load', filename
                _set = load(open(_dir+filename))
                topn.update([h for h in _set])
        topn = list(topn)
    
        # fix the llcnts to save time and make curve smooth
        print 'get llcnts...'
        topn = gen_fixlen_llcnts(topn, 5)
        dump(topn, open(_dir+'_nonadjacent_topn'+suffix, 'w'))

    else:
        print 'load', _dir
        topn = load(open(_dir))

    # find all correct hypotheses
    topn = list(topn)
    correct_set = set()

    for i in xrange(len(topn)):

        flag = True
        for k,v in topn[i].fixed_ll_counts.iteritems():                
            if len(k) < 2: 
                continue
            elif k[0] == 'a' and k[-1] in 'b':
                continue
            elif k[0] == 'c' and k[-1] in 'bd':
                continue
            elif k[0] == 'e' and k[-1] in 'bdf':
                continue
            flag=False
            break

        if flag: correct_set.add(i)

    print len(correct_set), 'of', len(topn), 'are correct'
    
    # get posterior
    w_list = range(2, 25, 1)
    amount_list = range(24, 144, 5)
    posterior_seq = []
    for i in xrange(len(w_list)):
        pool_size = w_list[i]
        language = LongDependency(max_length=pool_size)
        eval_data = [FunctionData(input=[], output={e: float(amount_list[i])/pool_size for e in language.str_sets})]

        for h in topn:
            h.likelihood_temperature = temperature
            h.compute_posterior(eval_data)

        Z = logsumexp([h.posterior_score for h in topn])
        
        
        prob = 0
        for i in xrange(len(topn)):
            if i in correct_set:
                prob += np.exp(topn[i].posterior_score - Z)
        print 'pool_size', pool_size, 'prob', prob
        posterior_seq.append([pool_size, prob])
        
        #debug
        _list = [h for h in topn]; _list.sort(key=lambda x: x.posterior_score, reverse=True)
        for i in xrange(3):
            print 'prob: ', np.exp(_list[i].posterior_score - Z),
            print h.fixed_ll_counts
            print _list[i]
        print '='*50; fff()

    dump(posterior_seq, open('nonadjacent_posterior_seq'+suffix,'w'))
 

def gen_fixlen_llcnts(topn, _len, num=2048):
    """
        force to generate llcnts with a fixed length
    """

    for h in topn:
        
        llcnts = Counter([h() for _ in xrange(2048)])
        for k,v in deepcopy(llcnts).iteritems():
            if len(k) != 5:
                del llcnts[k]
    
        now_num = float(sum(llcnts.values()))
        for k in deepcopy(llcnts).keys(): llcnts[k] *= (num/now_num)

        h.fixed_ll_counts = llcnts

    return topn






if __name__ == '__main__':

#    if options.MODE == 'staged_mk':
#        make_staged_seq2(jump=options.JUMP, temp=options.TEMP)
#    elif options.MODE == 'staged_plt':
#        get_kl_seq2(jump=options.JUMP, is_plot=options.PLOT, file_name=options.FILE)
#    elif options.MODE == 'nonadjacent_mk':
#        parse_nonadjacent(temperature=options.TEMP)
#    elif options.MODE == 'nonadjacent_plt':
#        dis_pos(jump=options.JUMP, is_plot=options.PLOT, file_name=options.FILE, axb_bound=options.AXB, x_bound=options.X)
#    elif options.MODE == 'parse_simple':
#        most_prob(options.FILE)
    if options.MODE == 'make_staged_posterior_seq':
        make_staged_posterior_seq(options.FILE, options.TEMP, options.LANG, options.DTYPE)
    elif options.MODE == 'parse_nonadjacent':
        parse_nonadjacent(options.FILE, options.TEMP)
    elif options.MODE == 'parse_plot':
        parse_plot(options.LANG, options.FINITE, is_plot=options.PLOT)
    elif options.MODE == 'only_plot':
        only_plot(options.FILE, options.STYPE)
    elif options.MODE == 'parse_cube':
        parse_cube(options.LANG, options.FINITE)
    elif options.MODE == 'cube2graph':
        cube2graph(options.FILE, options.PLOT)


    # seq = [[2  ,0.167012421	], [4  ,0.404128998	], [6  ,0.408503541], [8  ,0.346094762], [10 ,0.43615293], [12 ,0.995324843], [14 ,0.987169], [16 ,0.979347514], [18 ,0.983990461], [20 ,0.988215573	], [22 ,0.970604139	], [24 ,1]]
    # x, y = zip(*seq)
    # plt.plot(x, y)
    #
    # plt.ylabel('Weighted F-score')
    # plt.xlabel('Size of Data')
    # plt.title('Nonadjacent Dependency')
    # plt.show()

    # test_hypo_stat()
    #
    # cProfile.runctx("test_sto()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    # avg_seq_set = load(open('avg_seq_set_0829_022210'))
    # for avg_seq in avg_seq_set:
    #     for i in xrange(1, len(avg_seq)):
    #         avg_seq[i] = logsumexp([avg_seq[i], avg_seq[i-1]])
    #
    # staged, = plt.plot(range(12, 145, 30), avg_seq_set[0], label='staged')
    # normal, = plt.plot(range(12, 145, 30), avg_seq_set[1], label='normal')
    # uniform, = plt.plot(range(12, 145, 30), avg_seq_set[2], label='uniform')
    #
    # plt.legend(handles=[normal, staged, uniform])
    # plt.ylabel('KL-divergence')
    # plt.xlabel('data')
    # plt.show()

    # nonadjacent case
    # seq1 = []
    # seq2 = []
    # seq3 = []
    # seq4 = []
    #
    # f = open('123.txt')
    # for line in f:
    #     ad, nd, d, x = line.split()
    #     seq1.append(float(ad))
    #     seq2.append(float(nd))
    #     seq3.append(float(d))
    #     seq4.append(float(x))
    #
    # adjacent, = plt.plot(seq4, seq1, label='adjacent')
    # nonadjacent, = plt.plot(seq4, seq2, label='nonadjacent')
    # diff, = plt.plot(seq4, seq3, label='differences')
    #
    # plt.legend(handles=[adjacent, nonadjacent, diff], loc=5)
    # plt.ylabel('Weighted F-score')
    # plt.xlabel('Pool size')
    # plt.show()

    # seq1 = []
    # seq2 = []
    # seq3 = []
    #
    # f = open('123.txt')
    # for line in f:
    #     x, y = line.split()
    #     seq1.append(float(x))
    #     seq2.append(float(y))
    #
    # f = open('321.txt')
    # for line in f:
    #     x, y = line.split()
    #     seq3.append(float(y))
    #
    #
    # for i in xrange(1, len(seq1)):
    #     seq2[i] = (seq2[i] - seq2[i-1]) / (seq1[i] - seq1[i-1])
    # seq2[0] = 0
    #
    # for i in xrange(1, len(seq1)):
    #     seq3[i] = (seq3[i] - seq3[i-1]) / (seq1[i] - seq1[i-1])
    # seq3[0] = 0
    #
    # si, = plt.plot(np.log(seq1), seq2, label='SF')
    # rc, = plt.plot(np.log(seq1), seq3, label='RC')
    # plt.legend(handles=[si, rc], loc=5)
    # plt.axis([0, 5.6, 0, 1])
    # plt.ylabel('Cumulative KL-divergence')
    # plt.xlabel('Data size (log scale)')
    # plt.show()
    #
    # wfs = []
    # wp = []
    # wr = []
    # x = []
    # MAP_f = []
    #
    # fi = open('mod')
    # for line in fi:
    #     para = map(float, line.split())
    #     x.append(para[0])
    #     wfs.append(para[1])
    #     wp.append(para[2])
    #     wr.append((para[3]))
    #     MAP_f.append(para[4])
    #
    # plt.figure(figsize=(10, 6))
    # fig_wfs, = plt.plot(x, wfs, 'r', label='Weighted F-score')
    # fig_wp, = plt.plot(x, wp, 'r--', label='Weighted precision')
    # fig_wr, = plt.plot(x, wr, 'r-.', label='Weighted recall')
    # fig_map_f, = plt.plot(x, MAP_f, label='MAP hypothesis F-score')
    #
    # plt.ylabel('Scores')
    # plt.xlabel('Size of Data')
    # plt.legend(handles=[fig_wfs, fig_wp, fig_wr, fig_map_f], loc=5)
    # plt.axis([0, 30, 0, 1.05])
    # matplotlib.rcParams.update({'font.size': 16})
    # plt.show()
