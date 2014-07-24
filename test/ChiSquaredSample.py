from scipy.stats import chisquare
import random
import math


# sample chi-squared test, where we sample from a uniform distribution n times
def run_chisquare(n):
	num_possible_values = int(2*math.log(n))
	expected_counts = [(n + 0.0)/math.exp(i) for i in range(num_possible_values)]
	actual_counts = [0 for i in range(num_possible_values)]
	for i in range(n):
		random_value = random.randint(0, n-1)
		actual_counts[get_position(expected_counts, random_value)] += 1

	# run a chi-squared test
	print expected_counts, actual_counts
	return chisquare(actual_counts, f_exp=expected_counts)

# returns the position in the array where a random value should go
def get_position(expected_counts, random_value):
	total = 0
	for i in range(len(expected_counts)):
		total += expected_counts[i]
		if random_value < total: return i
	assert False, "Should not get here"

if __name__ == '__main__':
	for i in range(100,1000,100):
		print run_chisquare(i)
