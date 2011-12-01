CUDA_LIBDIR ?= /opt/cuda/lib64
DEBUG ?= -O3
TEXTURE_DIM ?= 1

NVCC = nvcc
prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include

ifeq ($(strip $(OSTYPE)),cygwin)
NVCCFLAGS = --compiler-bindir "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -Xcompiler /openmp
LFLAGS = -L"c:\Program Files\Microsoft SDKs\Windows\v7.1\Lib" -L.
RPATH =
LIBXGPU = libxgpu.dll
else
NVCCFLAGS = -Xcompiler -fopenmp -Xcompiler -D_REENTRANT -Xcompiler -Wall
LFLAGS = -L$(CUDA_LIBDIR) -L.
RPATH = -Xlinker -rpath,'$${ORIGIN}'
LIBXGPU = libxgpu.so
endif

NVCCFLAGS += $(DEBUG) -arch=sm_20 -Xptxas -abi=no -I. --ptxas-options=-v -prec-sqrt=false -Xcompiler -fPIC
NVCCFLAGS += -DTEXTURE_DIM=$(TEXTURE_DIM)

#NVCCFLAGS += -Xptxas -dlcm=cg # disable L1 cache

ifdef CUBE_COUNT_MODE
NVCCFLAGS += -DCUBE_COUNT_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_TIME_MODE
# FIXME? The test seems to fail on the timing run - why is this? - summation is doubled for some reason
NVCCFLAGS += -DCUBE_TIME_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_ASYNC_COUNT_MODE
NVCCFLAGS += -DCUBE_ASYNC_COUNT_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_ASYNC_TIME_MODE
NVCCFLAGS += -DCUBE_ASYNC_TIME_MODE
CUBE_OBJS = cube/cube.o
endif

# this sets the pipeline to run in an infinite loop - for power measurement
ifeq ($(strip $(POWER_LOOP)), yes)
NVCCFLAGS += -DPOWER_LOOP
endif

OBJS  = cuda_correlator.o

LIBXGPU_OBJS  = cuda_xengine.o
LIBXGPU_OBJS += omp_xengine.o
LIBXGPU_OBJS += cpu_util.o
LIBXGPU_OBJS += $(CUBE_OBJS)

# Each object file has a corresponding dependency file
DEPS = $(OBJS:.o=.d) $(LIBXGPU_OBJS:.o=.d)

all: cuda_correlator

cuda_correlator: $(OBJS) libxgpu.so
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LFLAGS) $(RPATH) -lxgpu

$(LIBXGPU): $(LIBXGPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LFLAGS) --shared

# A target to make all object files, but not link them
objs: $(OBJS) $(LIBXGPU_OBJS)

# Include dependencies after explicit rules (really anywhere after first
# explicit rule) so that they do not provide default target.
-include $(DEPS)

# These .{c,cc,cu} -> .o rules will generate dependency files as a side effect
# of the compilation.  These files are included elsewhere in this Makefile.
#
# NB Include dep files using "-include", see
#    http://make.paulandlesley.org/autodep.html#traditional for more info.

%.o: %.c
	@$(NVCC) $(NVCCFLAGS) -M -o $*.P $< && \
	cp $*.P $*.d && \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		  -e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d && \
	rm -f $*.P
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cc
	@$(NVCC) $(NVCCFLAGS) -M -o $*.P $< && \
	cp $*.P $*.d && \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		  -e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d && \
	rm -f $*.P
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cu
	@$(NVCC) $(NVCCFLAGS) -M -o $*.P $< && \
	cp $*.P $*.d && \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		  -e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d && \
	rm -f $*.P
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

install: $(LIBXGPU)
	mkdir -p $(includedir)
	cp xgpu.h $(includedir)
	mkdir -p $(libdir)
	cp $(LIBXGPU) $(libdir)

uninstall:
	rm -f $(includedir)/xgpu.h
	rm -f $(libdir)/$(LIBXGPU)

tags:
	ctags -R .

clean:
	rm -f ./cuda_correlator ./libxgpu.so
	rm -f $(OBJS) $(LIBXGPU_OBJS) cube/cube.o
	rm -f $(DEPS)
	rm -f tags

.PHONY : all objs tags clean install uninstall
