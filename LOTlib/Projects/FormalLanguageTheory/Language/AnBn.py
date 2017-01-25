
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar
from copy import deepcopy

class AnBn(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%sb', ['S'], 1.0)
        self.grammar.add_rule('S', 'ab',    None, 1.0)

    def terminals(self):
        return list('ab')

    @staticmethod
    def uniformdata(data):
        """
            Takes a FunctionData object and makes strings uniformly distributed.
            Used in skewed-frequency experiment
        """
        total_num = sum(data[0].output[k] for k in data[0].output)
        total_str = sum(1 for _ in data[0].output)
        num = float(total_num) / total_str

        for k in data[0].output:
            data[0].output[k] = num

        return data

    @staticmethod
    def stageddata(data, maxlen):
        """
            Takes a FunctionData object and remove strings that are longer than maxlen.
            Used in staged-input experiment
        """
        valid_num = 0.0
        total_num = 0.0
        dummy = deepcopy(data)

        for k in dummy[0].output:
            tmp = dummy[0].output[k]
            total_num += tmp
            if len(k) <= maxlen:
                valid_num += tmp
                continue
            data[0].output.pop(k, None)

        for k in data[0].output:
            data[0].output[k] *= total_num / valid_num

        return data



# just for testing
if __name__ == '__main__':
    language = AnBn()
    data = language.sample_data(10000)
    print data
    # print AnBn.uniformdata(deepcopy(data))
    # print AnBn.stageddata(deepcopy(data), 2)
    # print AnBn.stageddata(deepcopy(data), 4)
    # print AnBn.stageddata(deepcopy(data), 6)
