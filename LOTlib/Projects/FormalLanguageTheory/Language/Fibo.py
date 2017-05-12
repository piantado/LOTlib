
from LOTlib.Projects.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
from random import random

def fib(n):
    if n <= 1:
        return 1
    else:
        return fib(n-1)+fib(n-2)

class Fibo(FormalLanguage):
    """
    a^n : n is a fibonacci number
    """

    def __init__(self):
        self.grammar = None

    def terminals(self):
        return list('a')

    def sample_string(self): # fix that this is not CF
        n=0
        while random() < 0.5:
            n += 1
        return 'a'*fib(n)

        # just for testing
    def all_strings(self): # fix that this is not CF
        n=0
        while True:
            yield 'a'*fib(n)
            n += 1


if __name__ == '__main__':
    language = Fibo()
    print language.sample_data(10000)