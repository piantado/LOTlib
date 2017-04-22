# If in running you get 
# error while loading shared libraries: libcudart.so.2: cannot open shared object file: No such file or directory
# then you must add the cuda library to your path, as in export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64

CUDA_VERSION=7.0

CUDA_SAMPLES_INCLUDE=/usr/local/cuda/samples/common/inc/
CUDA_INCLUDE=/usr/local/cuda/lib64 
CUDA_LIB=/usr/local/cuda-$(CUDA_VERSION)/lib64

# NOTE: These options can be read off by running nvidia's (copyrighted) Makefile on some test examples
NVCC_FLAGS=-m64 -gencode arch=compute_30,code=sm_30 -use_fast_math 
# -use_fast_math forces everything to compile to __exp(x) versions, instead of slower and more accurate exp(x) versions...

HDF5_INCLUDE=/usr/include/hdf5/serial/
HDF_LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial

all: main.o
	g++ -m64 -o main main.o  -lstdc++ -lm -L$(CUDA_LIB) -lcudart -L$(HDF_LIB) -lhdf5
	rm main.o
	 
main.o: src/*.cu

	nvcc $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I. -I$(CUDA_SAMPLES_INCLUDE) -I$(HDF5_INCLUDE) -o main.o -c src/main.cu 
	
clean:
	rm -f main src/*.o *.o
