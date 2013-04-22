#!/bin/env python
# -*- coding: utf-8 -*-
"""Provides a map() interface to mpi4py.
License: MIT
Copywrite (c) 2012 Thomas Wiecki

Usage
*****

Create a python file (e.g. mpi_square.py):

from mpi4py import map_async
print map_async(lambda x, y: x**y, [1,2,3,4])

Then call the function with mpirun, e.g.:
mpi4run -n 4 mpi_square.py

https://github.com/twiecki/mpi4py_map/tree/master/mpi4py_map


TODO: UPDATE TO ALLOW MORE PROCESSES THAN SEQUENCES!

"""

import sys

from mpi4py import MPI

def map_async(function, sequence, args=None, debug=False):
    """Return a list of the results of applying the function in
    parallel (using mpi4py) to each element in sequence.

    :Arguments:
        function : python function
            Function to be called that takes as first argument an element of sequence.
        sequence : list
            Sequence of elements supplied to function.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        # Controller
        result = _mpi_controller(sequence, args=args, debug=debug)
        return result
    else:
        # Worker
        _mpi_worker(function, args=args, debug=debug)

def _mpi_controller(sequence, args=None, debug=False):
    """Controller function that sends each element in sequence to
    different workers. Handles queueing and job termination.

    :Arguments:
        sequence : list
            Sequence of elements supplied to the workers.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """
    rank = MPI.COMM_WORLD.Get_rank()
    assert rank == 0, "rank has to be 0."
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    process_list = range(1, MPI.COMM_WORLD.Get_size())
    workers_done = []
    results = []
    if debug: print "Data:", sequence

    queue = iter(sequence)

    if debug: print "Controller %i on %s: ready!" % (rank, proc_name)

    # Feed all queued jobs to the childs
    while(True):
        status = MPI.Status()
        # Receive input from workers.
        recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if debug: print "Controller: received tag %i from %s" % (status.tag, status.source)

        if status.tag == 1 or status.tag == 10:
            # tag 1 codes for initialization.
            # tag 10 codes for requesting more data.
            if status.tag == 10: # data received
                results.append(recv) # save back

            # Get next item and send to worker
            try:
                task = queue.next()
                # Send job to worker
                if debug: print "Controller: Sending task to %i" % status.source
                MPI.COMM_WORLD.send(task, dest=status.source, tag=10)

            except StopIteration:
                # Send kill signal
                if debug:
                    print "Controller: Task queue is empty"
                    print workers_done
                workers_done.append(status.source)
                MPI.COMM_WORLD.send([], dest=status.source, tag=2)

                # Task queue is empty
                if set(workers_done) == set(process_list):
                    break
                else:
                    continue

        # Tag 2 codes for a worker exiting.
        elif status.tag == 2:
            if debug:
                print 'Process %i exited, removing.' % status.source
                print 'Processes left over: ' + str(process_list)
            process_list.remove(status.source)

        else:
            print 'Unkown tag %i with msg %s' % (status.tag, str(recv))

        if len(process_list) == 0:
            if debug: print "All workers done."
            return results

def _mpi_worker(function, args=None, debug=False):
    """Worker that applies function to each element it receives from
    the controller.

    :Arguments:
        function : python function
            Function to be called that takes as first argument an
            element received from the controller.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """
    rank = MPI.COMM_WORLD.Get_rank()
    assert rank != 0, "rank is 0 which is reserved for the controller."
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()
    if debug: print "Worker %i on %s: ready!" % (rank, proc_name)

    # Send ready signal
    MPI.COMM_WORLD.send([{'rank': rank, 'name':proc_name}], dest=0, tag=1)

    # Start main data loop
    while True:
        # Wait for element
        if debug: print "Worker %i on %s: waiting for data" % (rank, proc_name)
        recv = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if debug: print "Worker %i on %s: received data, tag: %i" % (rank, proc_name, status.tag)

        if status.tag == 2:
            # Kill signal received
            if debug: print "Worker %i on %s: recieved kill signal" % (rank, proc_name)
            MPI.COMM_WORLD.send([], dest=0, tag=2)
            sys.exit(0)

        if status.tag == 10:
            # Call function on received element
            if debug: print "Worker %i on %s: Calling function %s with %s" % (rank, proc_name, function.__name__, recv)
            if args is None:
                result = function(recv)
            else:
                result = function(recv, *args)
            if debug: print("Worker %i on %s: finished one job" % (rank, proc_name))
            # Return result to controller
            MPI.COMM_WORLD.send(result, dest=0, tag=10)

def _power(x, y=2):
    return x**y

def test_map_async():
    import numpy as np
    result_parallel = map_async(_power, range(50), args=(2,), debug=True)
    result_serial = map(_power, range(50))
    assert np.all(result_serial == sorted(result_parallel)), "Parallel result does not match direct one."
    return sorted(result_parallel)

if __name__ == "__main__":
    result = test_map_async()
    print result
