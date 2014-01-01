
# Fancy memoize for interfacing with a lot -- from here -- http://ubuntuforums.org/showthread.php?t=1251060
class memoize(object):
	def __init__(self, func):
		self.func = func
		self.memoized = {}
		self.method_cache = {}
	def __call__(self, *args):
		return self.cache_get(self.memoized, args, lambda: self.func(*args))
	def __get__(self, obj, objtype):
		return self.cache_get(self.method_cache, obj,
		lambda: self.__class__(functools.partial(self.func, obj)))
	def cache_get(self, cache, key, func):
		try:
			return cache[key]
		except KeyError:
			cache[key] = func()
			return cache[key]
