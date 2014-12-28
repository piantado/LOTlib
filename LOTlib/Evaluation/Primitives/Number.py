from LOTlib.Evaluation.Eval import LOTlib_primitive

@LOTlib_primitive
def ends_in_(n, d):
    """Return true if number `n` ends with digit `d`, false otherwise. E.g. ends_in_(427, 7) == True"""
    if (n % 10) == d:
        return True
    else:
        return False

@LOTlib_primitive
def isprime_(n):
    """Is `n` a prime number?"""
    for a in range(2, int(n**0.5)+1):
        if n % a == 0:
            return False
    return True

@LOTlib_primitive
def primes_in_set_(A):
    def isPrime(a):
        for j in range(2, int(a**0.5)+1):
            if a % j == 0:
                return False
        return True

    return [n for n in A if isPrime(n)]

@LOTlib_primitive
def in_domain_(A, domain):
    return [n for n in A if (n <= domain)]

