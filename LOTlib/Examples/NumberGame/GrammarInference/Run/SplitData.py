

# TODO script to separate held out & training data
# TODO update command line arg so we can save held-out data to a file, then reload same file in MakeNGHs, Run





import csv
from optparse import OptionParser
from LOTlib.DataAndObjects import FunctionData


# ============================================================================================================
# Parsing command-line options

parser = OptionParser()

parser.add_option("-r", "--ratio",
                  dest="ratio", type="float", default="0.7",
                  help="Ratio of training : held-out data.")

parser.add_option("-f", "--filename",
                  dest="filename", type="string", default="MyData",
                  help="File name to save to (no extension).")




"""
For each row:
    row[0] is the FunctionData index number
    row[1] is either 'in' or 'out'

    if row[1] is 'in', row[2] is an input number
    if row[1] is 'out', row[2] is an output number, row[3] is # yes, row[4] is # no



"""

def csvToFunctionData(filename):
    with open(filename, mode='rb') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        ins = defaultdict()
        outs = defaultdict()

        # Fill `ins` and `outs` dictionaries
        for row in rows:
            if row[1] is 'in':
                ins[row[0]].append(row[2])
            if row[1] is 'out':
                outs[row[0]].append([row[2:]])

        # Fill FunctionData objects
        data_keys = set([row[0] for row in rows])
        data = {}
        for k in data_keys:
            data[k] = FunctionData(input=ins[k], output=outs[k])
        return data

def functionDataToCSV(filename, data):
    with open(filename, mode='wb') as f:
        writer = csv.writer(f)

        # For each data point
        for idx, d in enumerate(data):
            # Write a row for each input
            for i in d.input:
                writer.writerow([idx, 'in', i, None, None])
            # Write a row for each output
            for k in d.output.keys():
                o = d.output[k]
                writer.writerow([idx, 'out', k, o[0], o[1]])


def splitData(filename, ratio=0.7):

    data = csvToFunctionData(filename+'.csv')
    D = len(data) - 1

    idx = int(ratio * D)
    if idx == D:
        idx = D - 1

    with open(filename+'_main.csv', mode='wb') as fname:
        functionDataToCSV(fname, data[0:idx])

    with open(filename+'_hold.csv', mode='wb') as fname:
        functionDataToCSV(fname, data[idx:])

