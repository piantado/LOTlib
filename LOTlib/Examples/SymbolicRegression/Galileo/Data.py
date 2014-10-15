from LOTlib.DataAndObjects import FunctionData

# NOTE: these must be floats, else we get hung up on powers of ints
data_sd = 50.0
data = [
    FunctionData(input=[1000.], output=1500., ll_sd=data_sd),
    FunctionData(input=[828.], output=1340., ll_sd=data_sd),
    FunctionData(input=[800.], output=1328., ll_sd=data_sd),
    FunctionData(input=[600.], output=1172., ll_sd=data_sd),
    FunctionData(input=[300.], output=800., ll_sd=data_sd),
    FunctionData(input=[0.], output=0., ll_sd=data_sd) # added 0,0 since it makes physical sense.
]