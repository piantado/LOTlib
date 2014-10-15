__author__ = 'eric'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import Stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import Data
import Inference
import Specification
from Grammar import *


# grammar parameters


# Global parameters for inference
ALPHA = 0.9
NUM_ITERS = 10000
DATA = []


hypotheses = Inference.randomSample(grammar, DATA, num_iters=NUM_ITERS, alpha=ALPHA)


