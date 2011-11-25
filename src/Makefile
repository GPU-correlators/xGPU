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

ifdef CUBE_COUNT_MODE
NVCCFLAGS += -DCUBE_COUNT_MODE
endif

ifdef CUBE_TIME_MODE
# FIXME? The test seems to fail on the timing run - why is this? - summation is doubled for some reason
NVCCFLAGS += -DCUBE_TIME_MODE
endif

ifdef CUBE_ASYNC_COUNT_MODE
NVCCFLAGS += -DCUBE_ASYNC_COUNT_MODE
endif

ifdef CUBE_ASYNC_TIME_MODE
NVCCFLAGS += -DCUBE_ASYNC_TIME_MODE
endif

# this sets the pipeline to run in an infinite loop - for power measurement
ifeq ($(strip $(POWER_LOOP)), yes)
NVCCFLAGS += -DPOWER_LOOP
endif

OBJS  = cuda_correlator.o
OBJS += cuda_xengine.o
OBJS += omp_xengine.o
OBJS += cpu_util.o
OBJS += cube/cube.o

all: cuda_correlator

cuda_correlator: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LFLAGS)

# A target to make all object files, but not link them
objs: $(OBJS)

#cuda_correlator_old: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
#	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -o cuda_correlator $(LFLAGS)
#
#count_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
#	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_COUNT_MODE -o cuda_correlator $(LFLAGS)  
#
#time_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
#	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_TIME_MODE -o cuda_correlator $(LFLAGS) 
#
#async_count_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
#	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_COUNT_MODE -o cuda_correlator $(LFLAGS) 
#
#async_time_cuda_correlator: cuda_correlator.cu cuda_xengine.cu omp_xengine.cc cpu_util.cc
#	$(NVCC) $(NVCCFLAGS) cuda_correlator.cu cube/cube.c -DCUBE_ASYNC_TIME_MODE -o cuda_correlator $(LFLAGS) 

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cc
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tags:
	ctags -R .

clean:
	rm -f ./cuda_correlator
	rm -f ./*.o
	rm -f ./cube/cube.o

.PHONY : all objs tags clean
