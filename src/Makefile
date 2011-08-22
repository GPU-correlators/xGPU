ifeq ($(strip $(OSTYPE)),cygwin)
NVCC = nvcc
NVCCFLAGS = --compiler-bindir "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -L"c:\Program Files\Microsoft SDKs\Windows\v7.1\Lib" -Xcompiler /openmp
else
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -Xcompiler -fopenmp -Xcompiler -D_REENTRANT 
endif

NVCCFLAGS += -O3 -arch=sm_20 -Xptxas -abi=no -m32 -I. --ptxas-options=-v 

#NVCCFLAGS += -Xptxas -dlcm=cg # disable L1 cache

# this sets the pipeline to run in an infinite loop - for power measurement
ifeq ($(strip $(POWER_LOOP)), yes)
NVCCFLAGS += -DPOWER_LOOP
endif

all: cuda_correlator

cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -o cuda_correlator $(LFLAGS)

count_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_COUNT_MODE -o cuda_correlator $(LFLAGS)  

# FIXME? The test seems to fail on the timing run - why is this? - summation is doubled for some reason
time_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_TIME_MODE -o cuda_correlator $(LFLAGS) 

async_count_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_COUNT_MODE -o cuda_correlator $(LFLAGS) 

async_time_cuda_correlator: cuda_correlator.cu $(LIB) cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_TIME_MODE -o cuda_correlator $(LFLAGS) 

clean:
	rm -f ./cuda_correlator
