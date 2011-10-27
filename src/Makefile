CUDA_LIBDIR ?= /opt/cuda/lib64
DEBUG ?= -O3
TEXTURE_DIM ?= 1

ifeq ($(strip $(OSTYPE)),cygwin)
NVCC = nvcc
NVCCFLAGS = --compiler-bindir "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -L"c:\Program Files\Microsoft SDKs\Windows\v7.1\Lib" -Xcompiler /openmp
else
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fopenmp -Xcompiler -D_REENTRANT -L$(CUDA_LIBDIR)
endif

NVCCFLAGS += $(DEBUG) -arch=sm_20 -Xptxas -abi=no -I. --ptxas-options=-v -prec-sqrt=false
NVCCFLAGS += -DTEXTURE_DIM=$(TEXTURE_DIM)

#NVCCFLAGS += -Xptxas -dlcm=cg # disable L1 cache

# this sets the pipeline to run in an infinite loop - for power measurement
ifeq ($(strip $(POWER_LOOP)), yes)
NVCCFLAGS += -DPOWER_LOOP
endif

all: cuda_correlator

cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -o cuda_correlator $(LFLAGS)

count_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_COUNT_MODE -o cuda_correlator $(LFLAGS)  

# FIXME? The test seems to fail on the timing run - why is this? - summation is doubled for some reason
time_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_TIME_MODE -o cuda_correlator $(LFLAGS) 

async_count_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_COUNT_MODE -o cuda_correlator $(LFLAGS) 

async_time_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_TIME_MODE -o cuda_correlator $(LFLAGS) 

tags:
	ctags -R .

clean:
	rm -f ./cuda_correlator

.PHONY : tags
