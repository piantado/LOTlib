"""
        This makes a fancy sampler, meaning one that uses one kind of MultipleChainMCMC inside of another. 
        To do this, you have to subclass one and overwrite the make_sampler function

        NOTE: This is easy to run and see what happens with
            python Run.py  | grep "next_(WORD)"
        to just examine one partition
"""
from LOTlib import break_ctrlc

from LOTlib.Examples.Number.Model import grammar, make_h0, generate_data
data = generate_data(300)


from LOTlib.Inference.Samplers.ParallelTempering import ParallelTemperingSampler
from LOTlib.Inference.Samplers import PartitionMCMC

class MySampler(PartitionMCMC):
    """
    We make a PartitionMCMC that uses a ParallelTemperingSampler inside of it. This will make a ParallelTemperingSampler
    run in each partition
    """
    def make_sampler(self, make_h0, data, **kwargs):
        return ParallelTemperingSampler(make_h0, data,
                                        yield_only_t0=False, whichtemperature='acceptance_temperature', \
                                        temperatures=[1.0, 1.5, 2.0, 5.0, 10.0],
                                        **kwargs)


pmc = MySampler(grammar, make_h0, data, max_N=10, skip=0) # Initializer is just as for PartitionMCMC
for h in break_ctrlc(pmc):
    cps = pmc.chains[pmc.chain_idx] # the current partition sampler

    # Show the partition and the acceptance temperature
    print h.posterior_score, pmc.current_partition(), cps.chains[cps.chain_idx].acceptance_temperature, "\t", h
