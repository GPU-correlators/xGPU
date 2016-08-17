/*
  Simple cross-product, outputs in correct triangular form.

  - Coalescing memory access in all reads
  - No memory coalscing in writes (will be fixed)
  - Shared memory reads of type float2 to reduce global memory traffic
  - Each thread works on a 2x2 tile of data

  On a GTX 480 with >= 512 tiles this kernel achieve in excess of a
  teraflop.
 */

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <nvml.h>

#include "xgpu.h"
#include "xgpu_info.h"
#include "xgpu_version.h"
#include "cube/cube.h"
#include "power.h"

// whether we are writing the matrix back to device memory (used for benchmarking)
static int writeMatrix = 1;
// this must be enabled for this option to work though, slightly hurts performance
//#define WRITE_OPTION 

// System page size (used for rounding size passed to cudaHostRegister)
static long page_size = sysconf(_SC_PAGE_SIZE);

typedef struct XGPUInternalContextStruct {
  // Which device this context applies to
  int device;

  //memory pointers on the device
  ComplexInput *array_d[2];
  Complex *matrix_d;

  // used for overlapping comms and compute
  cudaStream_t streams[2];
  cudaEvent_t copyCompletion[2];
  cudaEvent_t kernelCompletion[2];

  // texture channel descriptor
  cudaChannelFormatDesc channelDesc;

  // Host input array that we allocated and should free
  ComplexInput * free_array_h;

  // Host input array that we registered and should unregister
  ComplexInput * unregister_array_h;

  // Whether xgpuSetHostInputBuffer has been called
  bool array_h_set;
  bool register_host_array;

  // Host output array that we allocated and should free
  Complex * free_matrix_h;

  // Host output array that we registered and should unregister
  Complex * unregister_matrix_h;

  // Whether xgpuSetHostOutputBuffer has been called
  bool matrix_h_set;
  bool register_host_matrix;
} XGPUInternalContext;

#define TILE_HEIGHT 8
#define TILE_WIDTH 8
#define NPOL 2

#define REG_TILE_NBASELINE ((NSTATION/2+1)*(NSTATION/4))

#ifndef FIXED_POINT
// texture declaration for FP32 reads
static texture<float2, 1, cudaReadModeElementType> tex1dfloat2;
static texture<float2, 2, cudaReadModeElementType> tex2dfloat2;
#else
#ifdef DP4A
// texture declaration for swizzled 8-bit fixed point reads
static texture<int2, 1, cudaReadModeElementType> tex1dchar4;
static texture<int2, 2, cudaReadModeElementType> tex2dchar4;
#else
// texture declaration for 8-bit fixed point reads
static texture<char2, 1, cudaReadModeNormalizedFloat> tex1dfloat2;
static texture<char2, 2, cudaReadModeNormalizedFloat> tex2dfloat2;
#endif
#endif

// array holding indices for which matrix we are doing the output to at a given iteration
#if (NPULSAR > 0)
static __device__ __constant__ unsigned char tIndex[PIPE_LENGTH*NFREQUENCY];
#endif

#define checkCudaError() do {                           \
    cudaError_t error = cudaGetLastError();		\
    if (error != cudaSuccess) {				\
      fprintf(stderr, "(CUDA) %s", cudaGetErrorString(error));	\
      fprintf(stderr, " (" __FILE__ ":%d)\n", __LINE__);		\
      return XGPU_CUDA_ERROR;						\
    }							\
  } while (0)

#ifdef TIME_CUDA_CALLS
#define CLOCK_GETTIME(clk_id, tp) clock_gettime(clk_id, tp)
#define PRINT_ELAPASED(f,t) printf("%s %ld ns\n", f, t)
#else
#define CLOCK_GETTIME(clk_id, tp)
#define PRINT_ELAPASED(f,t)
#endif

#include "kernel.cuh"

static XGPUInfo compiletime_info = {
  .npol =        NPOL,
  .nstation =    NSTATION,
  .nbaseline =   NBASELINE,
  .nfrequency =  NFREQUENCY,
  .ntime =       NTIME,
  .ntimepipe =   NTIME_PIPE,
#ifdef FIXED_POINT
  .input_type =  XGPU_INT8,
#else
  .input_type =  XGPU_FLOAT32,
#endif
#ifdef DP4A
  .compute_type = XGPU_INT8,
#else
  .compute_type = XGPU_FLOAT32,
#endif
  .vecLength  =  NFREQUENCY * NTIME * NSTATION * NPOL,
  .vecLengthPipe = NFREQUENCY * NTIME_PIPE * NSTATION * NPOL,
#if (MATRIX_ORDER == REGISTER_TILE_TRIANGULAR_ORDER)
  .matLength =   NFREQUENCY * ((NSTATION/2+1)*(NSTATION/4)*NPOL*NPOL*4) * (NPULSAR + 1),
#else
  // Matrix length is same for REAL_IMAG_TRIANGULAR_ORDER and TRIANGULAR_ORDER
  .matLength =   NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1),
#endif
  .triLength =   NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1),
  .matrix_order = MATRIX_ORDER,
  .shared_atomic_size = SHARED_ATOMIC_SIZE,
  .complex_block_size = COMPLEX_BLOCK_SIZE
};

// This stringification trick is from "info cpp"
#define STRINGIFY1(s) #s
#define STRINGIFY(s) STRINGIFY1(s)
static const char xgpu_version[] = STRINGIFY(XGPU_VERSION);

const char * xgpuVersionString()
{
  return xgpu_version;
}

// Populate XGPUInfo structure with compile-time parameters.
void xgpuInfo(XGPUInfo *pcxs)
{
  pcxs->npol           = compiletime_info.npol;
  pcxs->nstation       = compiletime_info.nstation;
  pcxs->nbaseline      = compiletime_info.nbaseline;
  pcxs->nfrequency     = compiletime_info.nfrequency;
  pcxs->ntime          = compiletime_info.ntime;
  pcxs->ntimepipe      = compiletime_info.ntimepipe;
  pcxs->input_type     = compiletime_info.input_type;
  pcxs->compute_type   = compiletime_info.compute_type;
  pcxs->vecLength      = compiletime_info.vecLength;
  pcxs->vecLengthPipe  = compiletime_info.vecLengthPipe;
  pcxs->matLength      = compiletime_info.matLength;
  pcxs->triLength      = compiletime_info.triLength;
  pcxs->matrix_order   = compiletime_info.matrix_order;
  pcxs->shared_atomic_size = compiletime_info.shared_atomic_size;
  pcxs->complex_block_size = compiletime_info.complex_block_size;
}

// Initialize the XGPU.  The device number is intentionally not part of the
// context because the device number needs to be maintained as part of the
// internal context (.e.g to ensure consistency with the device on which memory
// was allocated).
//
// TODO Cleanup as needed if returning due to error
int xgpuInit(XGPUContext *context, int device_flags)
{
  int error = XGPU_OK;

  CUBE_INIT();

  // Allocate internal context
  XGPUInternalContext *internal = (XGPUInternalContext *)malloc(sizeof(XGPUInternalContext));
  if(!internal) {
    // Uh-oh!
    return XGPU_OUT_OF_MEMORY;
  }
  context->internal = internal;
  internal->device = device_flags & XGPU_DEVICE_MASK;
  internal->array_h_set  = false;
  internal->matrix_h_set = false;
  internal->register_host_array  = true;
  internal->register_host_matrix = true;
  if( device_flags & XGPU_DONT_REGISTER_ARRAY ) {
	  internal->register_host_array = false;
  }
  if( device_flags & XGPU_DONT_REGISTER_MATRIX ) {
	  internal->register_host_matrix = false;
  }

  long long unsigned int vecLengthPipe = compiletime_info.vecLengthPipe;
  long long unsigned int matLength = compiletime_info.matLength;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("No CUDA devices found");
    exit(-1);
  }

  cudaDeviceProp deviceProp;
  for(int i=0; i<deviceCount; i++) {
    cudaGetDeviceProperties(&deviceProp, i);
    printf("Found device %d: %s\n", i, deviceProp.name);
  }

  cudaGetDeviceProperties(&deviceProp, internal->device);
  printf("Using device %d: %s\n", internal->device, deviceProp.name);

  //assign the device
  cudaSetDevice(internal->device);
  checkCudaError();

  // Setup input buffer
  internal->unregister_array_h = NULL;
  internal->free_array_h = NULL;
  if( internal->register_host_array ) {
	  // TODO error check
	  xgpuSetHostInputBuffer(context);
  }

  // Setup output buffer
  internal->unregister_matrix_h = NULL;
  internal->free_matrix_h = NULL;
  if( internal->register_host_matrix ) {
	  // TODO error check
	  xgpuSetHostOutputBuffer(context);
  }

  //allocate memory on device
  cudaMalloc((void **) &(internal->array_d[0]), vecLengthPipe*sizeof(ComplexInput));
  cudaMalloc((void **) &(internal->array_d[1]), vecLengthPipe*sizeof(ComplexInput));
  cudaMalloc((void **) &(internal->matrix_d), matLength*sizeof(Complex));
  checkCudaError();
  
  //clear out any previous values
  cudaMemset(internal->array_d[0], '\0', vecLengthPipe*sizeof(ComplexInput));
  cudaMemset(internal->array_d[1], '\0', vecLengthPipe*sizeof(ComplexInput));
  checkCudaError();

  // Clear device integration bufer
  error = xgpuClearDeviceIntegrationBuffer(context);
  if(error != XGPU_OK) {
    return error;
  }

  // create the streams
  for(int i=0; i<2; i++) cudaStreamCreate(&(internal->streams[i]));
  checkCudaError();

  // create the events
  for (int i=0; i<2; i++) {
    cudaEventCreateWithFlags(&(internal->kernelCompletion[i]), cudaEventDisableTiming);
    cudaEventCreateWithFlags(&(internal->copyCompletion[i]), cudaEventDisableTiming);
  }
  checkCudaError();

#ifndef FIXED_POINT
  internal->channelDesc = cudaCreateChannelDesc<float2>();
#else
#ifdef DP4A
  internal->channelDesc = cudaCreateChannelDesc<int2>();
#else
  internal->channelDesc = cudaCreateChannelDesc<char2>();
#endif // DP4A
#endif // FIXED_POINT

#if NPULSAR > 0
  unsigned char timeIndex[PIPE_LENGTH*NFREQUENCY];
  for (int tf=0; tf<PIPE_LENGTH*NFREQUENCY; tf++) timeIndex[tf] = 0;
  cudaMemcpyToSymbol(tIndex, timeIndex, PIPE_LENGTH*NFREQUENCY*sizeof(unsigned char), cudaMemcpyHostToDevice);

  checkCudaError();

  // check symbols are copied over
  unsigned char timeIndex2[PIPE_LENGTH*NFREQUENCY];
  cudaMemcpyFromSymbol(timeIndex2[t], tIndex[t], PIPE_LENGTH*NFREQUENCY*sizeof(unsigned char), cudaMemcpyDeviceToHost);  
  for (int tf=0; tf<PIPE_LENGTH*NFREQUENCY; tf++) {
    for (int f=0; f<NFREQUENCY; f++) 
      if (timeIndex[t][f] != timeIndex2[t][f]) 
	fprintf(stderr, "Index copy failed: t = %d, f = %d, original = %d, copy = %d\n", 
	       t, f, timeIndex[t][f], timeIndex2[t][f]);
  }
#endif

  // check whether texture dimensions are ok
#if TEXTURE_DIM == 2
#ifdef DP4A
  if((NFREQUENCY * NSTATION * NPOL > deviceProp.maxTexture2DLinear[0]) ||
     (NTIME_PIPE/4 > deviceProp.maxTexture2DLinear[1])) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#else
  if((NFREQUENCY * NSTATION * NPOL > deviceProp.maxTexture2DLinear[0]) ||
     (NTIME_PIPE > deviceProp.maxTexture2DLinear[1])) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#endif
#elif TEXTURE_DIM == 1
  // Surprisingly, this appears not to be a problem with 1D textures.  On a
  // GeForce GTX 580 (i.e. Fermi device), deviceQuery returns 65536 as
  // maxTexture1D, yet the default sizes use 10 * 256 * 2 * 100 * 2 == 1024000
  // bytes of 1D texture without any problems.  Perhaps the value of
  // maxTexture1D returned by cudaGetDeviceProperties is wrong?
#ifdef DP4A
  if (NFREQUENCY * NSTATION * NPOL * (NTIME_PIPE/4) > deviceProp.maxTexture1DLinear) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#else
  if (NFREQUENCY * NSTATION * NPOL * NTIME_PIPE > deviceProp.maxTexture1DLinear) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#endif
#endif 

#if !defined(CUBE_COUNT_MODE) && !defined(CUBE_ASYNC_COUNT_MODE) && !defined(CUBE_ASYNC_TIME_MODE)
  GPUmonitorInit(internal->device);
#endif

  return XGPU_OK;
}

// Clear the device integration buffer
int xgpuClearDeviceIntegrationBuffer(XGPUContext *context)
{
  long long unsigned int matLength = compiletime_info.matLength;

  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }
  //assign the device
  cudaSetDevice(internal->device);

  cudaMemset(internal->matrix_d, '\0', matLength*sizeof(Complex));
  checkCudaError();
  return XGPU_OK;
}

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

// Specify a new host input buffer.
int xgpuSetHostInputBuffer(XGPUContext *context)
{
#ifdef TIME_CUDA_CALLS
  struct timespec a, b;
#endif

  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }

  internal->array_h_set = true;

  //assign the device
  CLOCK_GETTIME(CLOCK_MONOTONIC, &a);
  cudaSetDevice(internal->device);
  CLOCK_GETTIME(CLOCK_MONOTONIC, &b);
  PRINT_ELAPASED("cudaSetDevice", ELAPSED_NS(a,b));

  if(internal->free_array_h) {
    CLOCK_GETTIME(CLOCK_MONOTONIC, &a);
    cudaFreeHost(internal->free_array_h);
    CLOCK_GETTIME(CLOCK_MONOTONIC, &b);
    PRINT_ELAPASED("cudaFreeHost", ELAPSED_NS(a,b));
  }
  if(internal->unregister_array_h) {
    CLOCK_GETTIME(CLOCK_MONOTONIC, &a);
    cudaHostUnregister(internal->unregister_array_h);
    CLOCK_GETTIME(CLOCK_MONOTONIC, &b);
    PRINT_ELAPASED("cudaHostUnregister", ELAPSED_NS(a,b));
  }

  if(context->array_h) {
    if( internal->register_host_array ) {
      // Register caller-allocated host memory with CUDA.
      // Round address down to nearest page_size boundary
      uintptr_t ptr_in = (uintptr_t)context->array_h;
      uintptr_t ptr_aligned = ptr_in - (ptr_in % page_size);
      // Compute length starting with compile time requirement
      size_t length = context->array_len * sizeof(ComplexInput);
      // TODO Verify that length is at least
      // "compiletime_info.vecLength*sizeof(ComplexInput)"

      // Add in any rounding that was done to the input pointer
      length += (ptr_in - ptr_aligned);
      // Round length up to next multiple of page size
      length = (length+page_size-1) / page_size * page_size;
#ifdef VERBOSE
      fprintf(stderr, "page aligned context->array_h = %p\n", ptr_aligned);
      fprintf(stderr, "length = %lx\n", length);
#endif
      CLOCK_GETTIME(CLOCK_MONOTONIC, &a);
      cudaHostRegister((void *)ptr_aligned, length, 0);
      CLOCK_GETTIME(CLOCK_MONOTONIC, &b);
      PRINT_ELAPASED("cudaHostRegister", ELAPSED_NS(a,b));
      internal->unregister_array_h = (ComplexInput *)ptr_aligned;
      internal->free_array_h = NULL;
      checkCudaError();
    }
    else {
      internal->unregister_array_h = NULL;
      internal->free_array_h = NULL;
    }
  } else {
    // allocate host memory
    context->array_len = compiletime_info.vecLength;
    CLOCK_GETTIME(CLOCK_MONOTONIC, &a);
    cudaMallocHost(&(context->array_h), context->array_len*sizeof(ComplexInput));
    CLOCK_GETTIME(CLOCK_MONOTONIC, &b);
    PRINT_ELAPASED("cudaMallocHost", ELAPSED_NS(a,b));
    internal->free_array_h = context->array_h;
    internal->unregister_array_h = NULL;
    checkCudaError();
  }

  // Init input_offset to 0
  context->input_offset = 0;

  return XGPU_OK;
}

// Specify a new host output buffer.
int xgpuSetHostOutputBuffer(XGPUContext *context)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }

  internal->matrix_h_set = true;

  //assign the device
  cudaSetDevice(internal->device);

  if(internal->free_matrix_h) {
    cudaFreeHost(internal->free_matrix_h);
  }
  if(internal->unregister_matrix_h) {
    cudaHostUnregister(internal->unregister_matrix_h);
  }

  if(context->matrix_h) {
    if( internal->register_host_matrix ) {
      // Register caller-allocated host memory with CUDA.
      // This requires that the caller allocated the memory properly vis-a-vis
      // the requirements of cudaHostRegister!
      // Round address down to nearest page_size boundary
      uintptr_t ptr_in = (uintptr_t)context->matrix_h;
      uintptr_t ptr_aligned = ptr_in - (ptr_in % page_size);
      // Compute length starting with compile time requirement
      size_t length = context->matrix_len * sizeof(Complex);
      // TODO Verify that length is at least
      // "compiletime_info.matLength*sizeof(Complex)"

      // Add in any rounding that was done to the input pointer
      length += (ptr_in - ptr_aligned);
      // Round length up to next multiple of page size
      length = (length+page_size-1) / page_size * page_size;
#ifdef VERBOSE
      fprintf(stderr, "page aligned context->matrix_h = %p\n", ptr_aligned);
      fprintf(stderr, "length = %lx\n", length);
#endif
      cudaHostRegister((void *)ptr_aligned, length, 0);
      internal->unregister_matrix_h = (Complex *)ptr_aligned;
      internal->free_matrix_h = NULL;
      checkCudaError();
    }
    else {
      internal->unregister_matrix_h = NULL;
      internal->free_matrix_h = NULL;
    }
  } else {
    // allocate host memory
    context->matrix_len = compiletime_info.matLength;
    cudaMallocHost(&(context->matrix_h), context->matrix_len*sizeof(Complex));
    internal->free_matrix_h = context->matrix_h;
    internal->unregister_matrix_h = NULL;
    checkCudaError();
  }

  // Init output_offset to 0
  context->output_offset = 0;

  return XGPU_OK;
}

// Free up the memory on the host and device
void xgpuFree(XGPUContext *context)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;

  if(internal) {
    //assign the device
    cudaSetDevice(internal->device);

    for(int i=0; i<2; i++) {
      cudaStreamDestroy(internal->streams[i]);
      cudaEventDestroy(internal->copyCompletion[i]);
      cudaEventDestroy(internal->kernelCompletion[i]);
    }

    if(internal->free_array_h) {
      cudaFreeHost(internal->free_array_h);
      context->array_h = NULL;
    }
    if(internal->unregister_array_h) {
      cudaHostUnregister(internal->unregister_array_h);
      context->array_h = NULL;
    }
    if(internal->free_matrix_h) {
      cudaFreeHost(internal->free_matrix_h);
      context->matrix_h = NULL;
    }
    if(internal->unregister_matrix_h) {
      cudaHostUnregister(internal->unregister_matrix_h);
      context->matrix_h = NULL;
    }

    cudaFree(internal->array_d[1]);
    cudaFree(internal->array_d[0]);
    cudaFree(internal->matrix_d);

    free(internal);
    context->internal = NULL;
  }

#if !defined(CUBE_COUNT_MODE) && !defined(CUBE_ASYNC_COUNT_MODE) && !defined(CUBE_ASYNC_TIME_MODE)
  GPUmonitorFree();
#endif

  CUBE_WRITE();
}

int xgpuCudaXengine(XGPUContext *context, int syncOp)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }

  // xgpuSetHostInputBuffer and xgpuSetHostOutputBuffer must have been called
  if( !internal->array_h_set || !internal->matrix_h_set ) {
    return XGPU_HOST_BUFFER_NOT_SET;
  }

  //assign the device
  cudaSetDevice(internal->device);

  ComplexInput **array_d = internal->array_d;
  cudaStream_t *streams = internal->streams;
  cudaEvent_t *copyCompletion = internal->copyCompletion;
  cudaEvent_t *kernelCompletion = internal->kernelCompletion;
  cudaChannelFormatDesc channelDesc = internal->channelDesc;

  // set pointers to the real and imaginary components of the device matrix
#ifndef DP4A
  float4 *matrix_real_d = (float4 *)(internal->matrix_d);
  float4 *matrix_imag_d = (float4 *)(internal->matrix_d + compiletime_info.matLength/2);
#else
  int4 *matrix_real_d = (int4 *)(internal->matrix_d);
  int4 *matrix_imag_d = (int4 *)(internal->matrix_d + compiletime_info.matLength/2);
#endif

  int Nblock = compiletime_info.nstation/min(TILE_HEIGHT,TILE_WIDTH);
  ComplexInput *array_load;
  ComplexInput *array_compute; 

  dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);
  //allocated exactly as many thread blocks as are needed
  dim3 dimGrid(((Nblock/2+1)*(Nblock/2))/2, compiletime_info.nfrequency);

  CUBE_ASYNC_START(ENTIRE_PIPELINE);

  // Need to fill pipeline before loop
  long long unsigned int vecLengthPipe = compiletime_info.vecLengthPipe;
  ComplexInput *array_hp = context->array_h + context->input_offset;
  // Only start the transfer once the kernel has completed processing input
  // buffer 0.  This is a no-op unless previous call to xgpuCudaXengine() had
  // SYNCOP_NONE or SYNCOP_SYNC_TRANSFER.
  cudaStreamWaitEvent(streams[0], kernelCompletion[0], 0);
  CUBE_ASYNC_COPY_CALL(array_d[0], array_hp, vecLengthPipe*sizeof(ComplexInput), cudaMemcpyHostToDevice, streams[0]);
  cudaEventRecord(copyCompletion[0], streams[0]); // record the completion of the h2d transfer
  checkCudaError();

  CUBE_ASYNC_START(PIPELINE_LOOP);

#ifdef POWER_LOOP
  for (int q=0; ; q++)
#endif
  for (int p=1; p<PIPE_LENGTH; p++) {
    array_compute = array_d[(p+1)%2];
    array_load = array_d[p%2];

    // Kernel Calculation
#if TEXTURE_DIM == 2
#ifndef DP4A
    cudaBindTexture2D(0, tex2dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE,
		      NFREQUENCY*NSTATION*NPOL*sizeof(ComplexInput));
#else
    cudaBindTexture2D(0, tex2dchar4, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE/4,
		      NFREQUENCY*NSTATION*NPOL*2*sizeof(char4));
#endif
#else
#ifndef DP4A
    cudaBindTexture(0, tex1dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*NTIME_PIPE*sizeof(ComplexInput));
#else
    cudaBindTexture(0, tex1dchar4, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*(NTIME_PIPE/4)*sizeof(int2));
#endif
#endif
    cudaStreamWaitEvent(streams[1], copyCompletion[(p+1)%2], 0); // only start the kernel once the h2d transfer is complete
    CUBE_ASYNC_KERNEL_CALL(shared2x2, dimGrid, dimBlock, 0, streams[1], 
			   matrix_real_d, matrix_imag_d, NSTATION, writeMatrix);
    cudaEventRecord(kernelCompletion[(p+1)%2], streams[1]); // record the completion of the kernel
    checkCudaError();

    // Download next chunk of input data
    cudaStreamWaitEvent(streams[0], kernelCompletion[p%2], 0); // only start the transfer once the kernel has completed
    CUBE_ASYNC_COPY_CALL(array_load, array_hp+p*vecLengthPipe, vecLengthPipe*sizeof(ComplexInput), cudaMemcpyHostToDevice, streams[0]);
    cudaEventRecord(copyCompletion[p%2], streams[0]); // record the completion of the h2d transfer
    checkCudaError();
  }

  CUBE_ASYNC_END(PIPELINE_LOOP);

  array_compute = array_d[(PIPE_LENGTH+1)%2];
  // Final kernel calculation
#if TEXTURE_DIM == 2
#ifndef DP4A
    cudaBindTexture2D(0, tex2dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE,
		      NFREQUENCY*NSTATION*NPOL*sizeof(ComplexInput));
#else
    cudaBindTexture2D(0, tex2dchar4, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE/4,
		      NFREQUENCY*NSTATION*NPOL*2*sizeof(char4));
#endif
#else
#ifndef DP4A
    cudaBindTexture(0, tex1dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*NTIME_PIPE*sizeof(ComplexInput));
#else
    cudaBindTexture(0, tex1dchar4, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*(NTIME_PIPE/4)*sizeof(int2));
#endif
#endif
  cudaStreamWaitEvent(streams[1], copyCompletion[(PIPE_LENGTH+1)%2], 0);
  CUBE_ASYNC_KERNEL_CALL(shared2x2, dimGrid, dimBlock, 0, streams[1], matrix_real_d, matrix_imag_d,
			 NSTATION, writeMatrix);

  if(syncOp == SYNCOP_DUMP) {
    checkCudaError();
    //copy the data back, employing a similar strategy as above
    CUBE_COPY_CALL(context->matrix_h + context->output_offset, internal->matrix_d, compiletime_info.matLength*sizeof(Complex), cudaMemcpyDeviceToHost);
    checkCudaError();
  } else if(syncOp == SYNCOP_SYNC_COMPUTE) {
    // Synchronize on the compute stream (i.e. wait for it to complete)
    cudaStreamSynchronize(streams[1]);
  } else {
      // record the completion of the kernel for next call
      cudaEventRecord(kernelCompletion[(PIPE_LENGTH+1)%2], streams[1]);
      checkCudaError();

      if(syncOp == SYNCOP_SYNC_TRANSFER) {
        // Synchronize on the transfer stream (i.e. wait for it to complete)
        cudaStreamSynchronize(streams[0]);
      }
  }

  CUBE_ASYNC_END(ENTIRE_PIPELINE);

  return XGPU_OK;
}
