from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--mode", dest="MODE", type="string", default='', help="")
(options, args) = parser.parse_args()


def human_experiment_stacked_bar(mode):
    """
        generate stacked bar for RC and SF case, assuming 12 blocks with each containing 12 data
    """
    N = 5
    mat = np.zeros((N, 12), dtype=np.float64)
    if mode == 'RC':
        mat[:, 0] = [31./5 for _ in xrange(N)]
    elif mode == 'SF':
        mat[:, 0] = [16., 8., 4., 2., 1.][:N]
    mat[:, 0] *= (12./31.)
    for i in xrange(1, mat.shape[1]):
        mat[:, i] = mat[:, 0] * (i+1)

    p_list = []
    width = 0.6
    ind = np.arange(1, 13)
    color_list = ['r', 'b', 'm', 'y', 'g']
    texture_list = ['/', '//', '\\', '*', 'o']
    for i in xrange(N):
        p = plt.bar(ind, mat[i, :], width=width, bottom=None if i == 0 else mat[:i, :].sum(0), color=color_list[i], hatch=texture_list[i])
        p_list.append(p)

    plt.xticks(ind + width/2., [str(i) for i in xrange(1, 13)])
    plt.yticks(np.arange(0, 145, 12))
    plt.xlabel('Blocks')
    plt.ylabel('Data Size')
    plt.legend([p[0] for p in p_list], ['$a^%db^%d$' % (x, x) for x in xrange(1, N+1)], loc=2)
    matplotlib.rcParams.update({'font.size': 24})
    plt.show()


def SI_stacked_bar():
    """
        generating this is quite different from the other two, we put it here
    """
    N = 5
    mat = np.zeros((N, 12), dtype=np.float64)

    mat[:, 0] = [16., 8., 4., 2., 1.][:N]
    mat[:, 0] *= (12./24.)
    for i in xrange(1, 4):
        mat[:, i] = mat[:, 0] * (i+1)

    mat[:, 0] = [16., 8., 4., 2., 1.][:N]
    mat[:, 0] *= (12./30.)
    for i in xrange(4, 8):
        mat[:, i] = mat[:, 0] * (i+1)

    mat[:, 0] = [16., 8., 4., 2., 1.][:N]
    mat[:, 0] *= (12./31.)
    for i in xrange(8, 12):
        mat[:, i] = mat[:, 0] * (i+1)

    p_list = []
    width = 0.6

    ind = np.arange(9, 13)
    color_list = ['r', 'b', 'm', 'y', 'g']
    texture_list = ['/', '//', '\\', '*', 'o']
    for i in xrange(5):
        p = plt.bar(ind, mat[i, 8:12], width=width, bottom=None if i == 0 else mat[:i, 8:12].sum(0), color=color_list[i], hatch=texture_list[i])
        p_list.append(p)

    plt.xticks(np.arange(1, 13) + width/2., [str(i) for i in xrange(1, 13)])
    plt.yticks(np.arange(0, 145, 12))
    plt.xlim(0, 14)
    plt.ylim(0, 155)
    plt.xlabel('Blocks')
    plt.ylabel('Data Size')
    plt.legend([p[0] for p in p_list], ['$a^%db^%d$' % (x, x) for x in xrange(1, N+1)], loc=2)

    ind = np.arange(1, 5)
    color_list = ['r', 'b', 'm', 'y', 'g']
    texture_list = ['/', '//', '\\', '*', 'o']
    for i in xrange(2):
        p = plt.bar(ind, mat[i, :4], width=width, bottom=None if i == 0 else mat[:i, :4].sum(0), color=color_list[i], hatch=texture_list[i])
        p_list.append(p)

    ind = np.arange(5, 9)
    color_list = ['r', 'b', 'm', 'y', 'g']
    texture_list = ['/', '//', '\\', '*', 'o']
    for i in xrange(4):
        p = plt.bar(ind, mat[i, 4:8], width=width, bottom=None if i == 0 else mat[:i, 4:8].sum(0), color=color_list[i], hatch=texture_list[i])
        p_list.append(p)



    matplotlib.rcParams.update({'font.size': 24})
    plt.show()

if __name__ == '__main__':

    # SI_stacked_bar()

    human_experiment_stacked_bar(options.MODE)