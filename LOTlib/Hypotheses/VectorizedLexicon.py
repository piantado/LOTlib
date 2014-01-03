import Hypothesis

class VectorizedLexicon(Hypothesis):
	"""
		This is a Lexicon class that stores *only* indices into my_finite_trees, and is designed for doing gibbs,
		sampling over each possible word meaning. It requires running EnumerateTrees.py to create a finite set of 
		trees, and then loading them here. Then, all inferences etc. are vectorized for super speed.
		
		This requires a bunch of variables (the global ones) from run, so it cannot portably be extracted. But it can't
		be defined inside run or else it won't pickle correctly. Ah, python. 
		
		NOTE: This takes a weirdo form for the likelihood function
	"""
	
	def __init__(self, target_words, finite_trees, priorlist, word_idx=None, ALPHA=0.75, PALPHA=0.75):
		Hypothesis.__init__(self)
		
		self.__dict__.update(locals())
		
		self.N = len(target_words)
		
		if self.word_idx is None: 
			self.word_idx = np.array( [randint(0,len(self.priorlist)-1) for i in xrange(self.N) ])
		
	def __repr__(self): return str(self)
	def __eq__(self, other): return np.all(self.word_idx == other.word_idx)
	def __hash__(self): return hash(tuple(self.word_idx))
	def __cmp__(self, other): return cmp(str(self), str(other))
	
	def __str__(self): 
		s = ''
		aw = self.target_words
		for i in xrange(len(self.word_idx)):
			s = s + aw[i] + "\t" + str(self.finite_trees[self.word_idx[i]]) + "\n"
		s = s + '\n'
		return s
	
	def __copy__(self):
		return VectorizedLexicon(self.target_words, self.finite_trees, self.priorlist, word_idx=np.copy(self.word_idx), ALPHA=self.ALPHA, PALPHA=self.PALPHA)
	
	def enumerative_proposer(self, wd):
		for k in xrange(len(self.finite_trees)):
			new = copy(self)
			new.word_idx[wd] = k
			yield new
			
	def compute_prior(self):
		self.prior = sum([self.priorlist[x] for x in self.word_idx])
		self.lp = self.prior + self.likelihood
		return self.prior
	
	def compute_likelihood(self, data):
		"""
			Compute the likelihood on the data, super fast
			
			These use the global variables defined by run below.
			The alternative is to define this class inside of run, but that causes pickling errors
		"""
		
		response,weights,uttered_word_index = data # unpack the data
		
		# vector 0,1,2, ... number of columsn
		zerothroughcols = np.array(range(len(uttered_word_index)))
		
		r = response[self.word_idx] # gives vectors of responses to each data point
		w = weights[self.word_idx]
		
		if r.shape[1] == 0:  # if no data
			self.likelihood = 0.0
		else:
			
			true_weights = ((r > 0).transpose() * w).transpose().sum(axis=0)
			met_weights  = ((np.abs(r) > 0).transpose() * w).transpose().sum(axis=0)
			all_weights = w.sum(axis=0)
			rutt = r[uttered_word_index, zerothroughcols] # return value of the word actually uttered
			
			## now compute the likelihood:
			lp = np.sum( np.log( self.PALPHA*self.ALPHA*weights[self.word_idx[uttered_word_index]]*(rutt>0) / (1e-5 + true_weights) + \
					self.PALPHA*( (true_weights>0)*(1.0-self.ALPHA) + (true_weights==0)*1.0) * weights[self.word_idx[uttered_word_index]] * (np.abs(rutt) > 0) / (1e-5 + met_weights) + \
					( (met_weights>0)*(1.0-self.PALPHA) + (met_weights==0)*1.0 ) * weights[self.word_idx[uttered_word_index]] / (1e-5 + all_weights)))
				
			self.likelihood = lp
		self.lp = self.likelihood+self.prior
		return self.likelihood 
		
	def propose(self):
		print "*** Cannot propose to VectorizedLexicon"
		assert False