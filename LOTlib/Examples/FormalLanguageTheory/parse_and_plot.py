import matplotlib.pyplot as plt

data_size = []
posterior_prob = []
posterior_score = []
prior = []
likelihood = []
num_of_string = []
hypotheses = []
precision = []
recall = []


def read_data(input):
    f = open(input, 'r')
    global data_size
    global posterior_prob
    global posterior_score
    global prior
    global likelihood
    global num_of_string
    global hypotheses
    global precision
    global recall

    for line in f:
        set = line.split()
        data_size.append(int(set[0]))
        posterior_prob.append(float(set[1]))
        posterior_score.append(float(set[2]))
        prior.append(float(set[3]))
        likelihood.append(float(set[4]))
        num_of_string.append(int(set[5]))
        hypotheses.append(set[6])
        precision.append(float(set[7]))
        recall.append(float(set[8]))



read_data('./out/out')
plt.plot()

