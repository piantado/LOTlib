"""
    The exception we throw for all problems in Evaluation
"""

class EvaluationException(Exception):
    pass

class TooBigException(EvaluationException):
    pass

class RecursionDepthException(EvaluationException):
    pass