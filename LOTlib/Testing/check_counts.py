"""
	Check some counts according to a chi-squared statistic. 
	
	We can use this to see if sampling counts, etc. are what they should be.
"""


def check_counts( obj2counts, expected, threshold=0.001, verbose=False):
	"""
		Here, obj2counts is a dictionary mapping each thing to a count
		expected is a *function* that takes an object and hands back its expected counts (normalized)
	"""
	from scipy.stats import chisquare
	
	objects = obj2counts.keys()
	
	actual_counts   = map(lambda o: float(obj2counts[o]), objects)
	N = sum(actual_counts)
	expected_counts = map(lambda o: float(expected(o)*N), objects)
	
	chi, p = chisquare(f_obs=actual_counts, f_exp=expected_counts)

	if verbose:
		print "# Chi squared gives chi=%f, p=%f" % (chi,p)
		if p < threshold:
			assert "# *** SIGNIFICANT DEVIATION FOUND IN P"
	
	assert p > threshold, "*** Chi squared test fail with chi=%f, p=%f" % (chi,p)
	
	return True
	
		