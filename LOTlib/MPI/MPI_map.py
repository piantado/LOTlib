# -*- coding: utf-8 -*-
"""
    TODO: Maybe we have to have MPI_done wait until the children have officially exited?

    The MPI interface for python mapping

    Slave processes are caught either by calling capture_slaves(), or the first time you call MPI_map. In either case, they wait around indefinitely until told to exit

    It is not currently possible to return slaves from MPI_map

    TODO: Check that we shouldn't be using Size-1 in order to keep track of the master process

"""


from mpi4py import MPI
import random
from ProgressBar import draw_progress
from ParallelBufferedIO import ParallelBufferedIO
import sys

# Which dumps to use?
from pickle import dumps # No lambda support
#from cloud.serialization.cloudpickle import dumps

# Tags for message passing
SYNCHRONIZE_TAG = 0x2
EXIT_TAG = 0x1
RUN_TAG = 0x0
MASTER_PROCESS = 0 # what process is the master node?

SLAVE_RETURN = None # what do slaves return when they complete?

# Each thread gets its rank and size here:
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
out = None

def get_size():
    global size
    return size
def get_rank():
    global rank
    return rank

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make sure we always finalize when we exit, or else all hell breaks loose, with loose MPI threads hanging around
# IT is much better to call MPI_done(), but this catches things just in case we don't
import atexit # for running things at exit
def myexit():

    # Tell children to end
    if is_master_process(): MPI_done()

    ## TODO: By telling children to exit, it may take a while for them to finish..

    global out
    if out is not None: out.close() # close this -- meaning the subprocess is told to stop

    if not MPI.Is_finalized():
        MPI.Finalize()
atexit.register(myexit)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def is_master_process():
    """
    You are master if you are size=0 (not in MPI) or the master process
    """
    return (size==0) or (rank == MASTER_PROCESS)


def synchronize_variable(f):
    """
        Evaluate f on the head node to yield a variable value, and then send that to everything.
        So in our code, we can say

        y = synchronize_variable( lambda : random.random())

        this only evals the function on the master_node, and then synchronizes to everyone else
    """
    if size ==1:
        return f()
    elif rank == MASTER_PROCESS:
        ret = f() # evaluate
        for i in xrange(size): # send this to each process
            if i != MASTER_PROCESS: comm.send( ret, dest=i, tag=SYNCHRONIZE_TAG)
        return ret
    else:
        while True:
            # test for the exit code
            if comm.Iprobe(source=MASTER_PROCESS, tag=SYNCHRONIZE_TAG):
                return comm.recv(source=MASTER_PROCESS, tag=SYNCHRONIZE_TAG)


def worker_process(outfile=None):
    """
        This implements a worker process who is sitting to receive
    """
    global out # make this global so that myexit can call it
    assert rank != MASTER_PROCESS, "Hmm: worker should not be a master process" # better not have master processes in here!

    if outfile is not None:
        out = ParallelBufferedIO(outfile)

    while True:

        # test for the exit code
        if comm.Iprobe(source=MASTER_PROCESS, tag=EXIT_TAG):
            comm.recv(source=MASTER_PROCESS, tag=EXIT_TAG)
            sys.exit(0) # This thread is done

        # test for a function to evaluate
        if comm.Iprobe(source=MASTER_PROCESS, tag=RUN_TAG):
            f, i, a = comm.recv(source=MASTER_PROCESS, tag=RUN_TAG) # get our next job
            #print "#", rank, " received ", i, a
            r = f(*a)

            if outfile is not None:
                out.write(*r) # non-blocking write to a file if we want it

            # send a message that we've finished
            comm.send([i, r], dest=MASTER_PROCESS, tag=RUN_TAG)

    if outfile is not None:
        out.close()

def capture_slaves(outfile=None):
    """
        You can call this on an MPI process, to start the slave processign requests.
        It is called defaultly the first time you call MPI_map, but it can be called earlier in the program if you have some data processing that not all slaves should do

        outfile - an output file for the slave to print to. This is thread-safe
    """
    if rank != MASTER_PROCESS:
        worker_process(outfile=outfile)

def MPI_done():

    assert is_master_process()

    # Tell all to exit from this map (Not exit overall)
    for i in range(1,size):
        #dprinterr(25, "# Master calling exit on ", i)
        comm.send(None, dest=i, tag=EXIT_TAG)


def MPI_unorderedmap(f, generator):
    """
        Evaluates f on each element of a generator g, in no particular order. This is much simpler

        NOTE: This MUST be used in the context of a generator:

        for g in MPI_unorderedmap(f,gen):
            pass

        for instance.
    """
    capture_slaves()
    assert is_master_process()
    global size
    if size == 1:
        print "# *** NOTE: 'MPI_unorderedmap' running as 'map' since size=1!"
        for g in iter(generator):
            yield f(*g)

        raise StopIteration
    else:
        generator = iter(generator) # just so we can pass in lists
        running = [False]*size  # which processes are running?

        gotAllArgs = False
        try:
            while (not gotAllArgs) or any(running):
                # print gotAllArgs, running
                for i in xrange(1,size):
                    if comm.Iprobe(source=i, tag=RUN_TAG): # test for a message

                        ri, r = comm.recv(source=i, tag=RUN_TAG) # get the message

                        yield r

                        running[i] = False # so we send it a new job

                    if (not running[i]) and not gotAllArgs: # give it a job

                        # Get the next argument
                        try:
                            arg = generator.next()

                            #print "# Sending ", arg, " to ", i
                            comm.send([f, None, arg], dest=i, tag=RUN_TAG)

                            running[i] = True

                        except StopIteration:
                            gotAllArgs = True

        except Exception as e:
            print >>sys.stderr, "EXCEPTION IN MPI_unorderedmap. Shutting down", e
            MPI_done()  # shut down everyone (on, e.g., interrupt, etc)
            raise


        raise StopIteration



# Let's make a new kind, where we spawn up to the length in order to process, each time we see an MPI_map
#http://mpi4py.scipy.org/docs/usrman/tutorial.html
def MPI_map(f, args, outfile=None, mpi_done=False, yieldfrom=False, progress_bar=True):
    """
        Execute, in parallel, a function on each argument, and return the list [x1, f(x1)], [x2, f(x2)].

        f -- the function. Must be defined using "def" (not lambda) or else the slaves can't see it
        args -- a list of arguments to apply f to
        outfile - should reuslts be printed nicely in parallel to outfile?
        mpi_done - if True, we tell all subprocesses to die. This is handy if you only have one MPI_map, and
        therefore don't need the processes again
        # yieldfrom - if True, we yield each individual response NOTE TODO: CURRENTLY NOT IMPLEMENTED
    """
    global size

    if size == 1:
        print "# *** NOTE: 'MPI_map' running as 'map' since size=1!"
        return map(lambda x: f( *x), args)

    # calling this is a sink that all slave processes fall into, waiting
    capture_slaves(outfile)
    assert is_master_process()

    arglen = len(args)
    started_count = 0 # how many things do we need to get back?
    completed_count = 0
    running = [False]*size  # which processes are running?
    ind = range(arglen)

    #if not yieldfrom:
    ret = [None]*arglen # the return values

    # Now only the master process survives:
    try:
        if progress_bar: draw_progress(float(completed_count)/float(arglen)) # Just to draw it to start

        while completed_count < arglen:

            #print size, arglen, completed_count, range(1,min(size,arglen+1),1)
            for i in range(1,min(size,arglen+1),1): # run at most the number of arguments in parallel
                #print "#", i, running[i], size, arglen

                if (not running[i]) and (started_count < arglen):
                    #print "# Main sending ", args[ind[started_count]], " to ", i
                    comm.send([f, ind[started_count],  args[ind[started_count]] ], dest=i, tag=RUN_TAG)
                    started_count += 1
                    running[i] = True

                if comm.Iprobe(source=i, tag=RUN_TAG): # test for a message
                    completed_count += 1 # we've gotten back one more
                    if progress_bar: draw_progress(float(completed_count)/float(arglen))

                    ri, r = comm.recv(source=i, tag=RUN_TAG) # get the message
                    #if yieldfrom: yield [ args[ind[ri]] , r] # return the args, and the response
                    #else:         ret[ri] = r # save it
                    ret[ri] = r # save it
                    running[i] = False # so we send it a new job

        # if we don't need the slave processes anymore
        if mpi_done: MPI_done()

    except Exception as e:
        print >>sys.stderr, "EXCEPTION IN MPI_map. Shutting down", e
        MPI_done() # shut down everyone (on, e.g., interrupt, etc)
        raise e

    if progress_bar: print >>sys.stderr, "\n" # since we drew the progress bar!

    #if not yieldfrom:
    return ret


if __name__=="__main__":
    # mpiexec -n 36 python test-mpi.py

    #  a somewhat hard math problem
    def f(i): return [i, i ** 15000]

    ## Run a few maps:
    r = MPI_map(f, map(lambda x: x, range(1500)))
    assert len(r) == 1500

    #MPI_map(f, map(lambda x: [x], range(1500)))

    for g in MPI_unorderedmap(f, map(lambda x: [x], range(1500))):
        #print g
        pass

    MPI_map(f, map(lambda x: [x+x], range(1500)))

    MPI_map(f, map(lambda x: [x*x], range(1500)))

    # And give a default outfile:
    MPI_map(f, map(lambda x: [x], range(500)), outfile="/tmp/mpitest.txt")

    sys.exit(0)
