NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -g -O3 $(INC) -arch=sm_11 --ptxas-options=-v -Xcompiler -fopenmp -Xcompiler -D_REENTRANT -Xcompiler -D_FILE_OFFSET_BITS=64 -Xptxas -abi=no -m64 #-Xptxas -dlcm=cg # disable L1 cache

all: cuda_correlator

cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -o cuda_correlator $(LFLAGS)  

count_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_COUNT_MODE -o cuda_correlator $(LFLAGS) 

# FIXME? The test seems to fail on the timing run - why is this?
time_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_TIME_MODE -o cuda_correlator $(LFLAGS) 

async_count_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_COUNT_MODE -o cuda_correlator $(LFLAGS) 

async_time_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_TIME_MODE -o cuda_correlator $(LFLAGS) 

clean:
	rm -f ./cuda_correlator
