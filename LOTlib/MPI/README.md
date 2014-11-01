
A friendly wrapper for mpi4py that allows maps to run on each mpi thread. 

Importing MPI_Map.MPI_Map will allow easy parallel mapping:

	y = MPI_map(f, range(1500))

where each job is sent to a separate core, when started like:

	mpirun -n 4 python MPI_map.py


To time on a simple test:

time mpirun -n 2 python MPI_map.py

time mpirun -n 10 python MPI_map.py

REQUIREMENTS
=============

	mpi4py (apt-get install works best when mpich2 is also apt-get installed)
	multiprocessing (for parallel buffered i/o on the host)
	lockfile
	
INSTALLATION:
=============

This library requires mpi4py and mpich. I often have to compile these from source, using the configure --enable-shared flag in mpich3 in order to get it to work. 
