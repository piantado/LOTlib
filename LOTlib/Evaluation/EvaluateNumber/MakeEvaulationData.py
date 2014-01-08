TARGET_FILE = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpirun-Dec2013.pkl" # load a small file. The large one is only necessary if we want the "correct" target likelihood and top N numbers; if we just look at Z we don't need it!
DATA_FILE = "data/evaluation-data.pkl"

TARGET_SAMPLES = 50000


# make and save the data
data = generate_data(DATA_SIZE)
pickle_save(data, DATA_FILE)

initial_hyp = NumberExpression(G)

q = FiniteBestSet(10000)
for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, TARGET_SAMPLES, skip=0):
	q.push(h, h.lp)
q.save(TARGET_FILE)