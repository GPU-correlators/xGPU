#ifndef CUBE_H
#define CUBE_H

// maximum number of kernels CUBE can track
#define CUBE_KERNEL_MAX 100
#define ulonglong unsigned long long int

#if defined(__cplusplus)
extern "C" {
#endif

  extern unsigned int CUBE_nCall; // number of times to call each kernel

  extern unsigned int CUBE_nKernel;
  extern ulonglong CUBE_Flops[CUBE_KERNEL_MAX];
  extern ulonglong CUBE_Bytes[CUBE_KERNEL_MAX];
  extern float CUBE_Times[CUBE_KERNEL_MAX];
  extern char *CUBE_Names[CUBE_KERNEL_MAX];

  void CUBE_Init(void);
  void CUBE_Write_Flops(void);
  void CUBE_Write_Benchmark(void);
  int CUBE_Get_Index(const char *);

#if defined(__cplusplus)
}
#endif

// All the C-preprocessor "magic" is below this

#define CUBE_FLOPS 1000
#define CUBE_TIME  2000

#if defined(CUBE_COUNT_MODE)
#define CUBE_MODE CUBE_FLOPS // enable for intial flops counting
#elif defined(CUBE_TIME_MODE)
#define CUBE_MODE CUBE_TIME  // enable for final benchmark
#endif

#ifndef CUBE_NCALL
#define CUBE_NCALL 50
#endif

// Counting macro definitions

#define CUBE_KERNEL_CALL_COUNT(kernel, grid, threads, shared, ...)	\
  {									\
    ulonglong *d_CUBE_Flops = 0;					\
    ulonglong *d_CUBE_Bytes = 0;					\
    cudaMalloc( (void**)&d_CUBE_Flops, sizeof(ulonglong));		\
    cudaMemset( d_CUBE_Flops, 0, sizeof(ulonglong));			\
    cudaMalloc( (void**)&d_CUBE_Bytes, sizeof(ulonglong)) ;		\
    cudaMemset( d_CUBE_Bytes, 0, sizeof(ulonglong)) ;			\
    kernel <<< grid , threads, shared >>> ( __VA_ARGS__ , d_CUBE_Flops, d_CUBE_Bytes); \
    ulonglong h_CUBE_Flops, h_CUBE_Bytes;				\
    cudaMemcpy(&h_CUBE_Flops, d_CUBE_Flops, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaMemcpy(&h_CUBE_Bytes, d_CUBE_Bytes, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaFree( d_CUBE_Flops );						\
    cudaFree( d_CUBE_Bytes );						\
    int index = CUBE_Get_Index(#kernel);				\
    CUBE_Flops[index] += h_CUBE_Flops;					\
    CUBE_Bytes[index] += h_CUBE_Bytes;					\
  }

#define CUBE_KERNEL_COUNT(kernel, ...)					\
  __global__ void kernel(__VA_ARGS__, ulonglong *CUBE_total_flops, ulonglong *CUBE_total_bytes)

#define CUBE_DEVICE_CALL_COUNT(function, ...)	\
  function(__VA_ARGS__, CUBE_flops, CUBE_bytes)

#define CUBE_DEVICE_COUNT(rtn_type, function, ...)			\
  __device__ rtn_type function(__VA_ARGS__, ulonglong &CUBE_flops, ulonglong &CUBE_bytes)


// Timing macro definitions

#define CUBE_KERNEL_CALL_TIME(kernel, grid, threads, shared, ...)	\
  {									\
    cudaEvent_t start, end;						\
    cudaEventCreate(&start);						\
    cudaEventCreate(&end);						\
    cudaEventSynchronize(start);					\
    cudaEventRecord(start, 0);						\
    for (int i=0; i<CUBE_nCall; i++)					\
      kernel <<< grid , threads, shared >>> ( __VA_ARGS__ );		\
    cudaEventRecord(end, 0);						\
    cudaEventSynchronize(end);						\
    float runTime;							\
    cudaEventElapsedTime(&runTime, start, end);				\
    cudaEventDestroy(start);						\
    cudaEventDestroy(end);						\
    int index = CUBE_Get_Index(#kernel);				\
    CUBE_Times[index] += runTime / CUBE_nCall;				\
  }
  



// Default macro definitions

#define CUBE_KERNEL_CALL_DEFAULT(kernel, grid, threads, shared, ...)	\
  kernel <<< grid , threads, shared >>>(__VA_ARGS__);

#define CUBE_KERNEL_DEFAULT(kernel, ...)	\
  __global__ void kernel(__VA_ARGS__)

#define CUBE_DEVICE_CALL_DEFAULT(function, ...)	\
  function(__VA_ARGS__)

#define CUBE_DEVICE_DEFAULT(rtn_type, function, ...)	\
  __device__ rtn_type function(__VA_ARGS__)


// set macros according to mode

#if (CUBE_MODE == CUBE_FLOPS) // Count flops and bytes

#define CUBE_KERNEL_CALL CUBE_KERNEL_CALL_COUNT
#define CUBE_KERNEL CUBE_KERNEL_COUNT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_COUNT
#define CUBE_DEVICE CUBE_DEVICE_COUNT

// create flop and byte counters

#define CUBE_START				\
  ulonglong CUBE_flops = 0;			\
  ulonglong CUBE_bytes = 0;			

#define CUBE_ADD_FLOPS(x) CUBE_flops += x;
#define CUBE_ADD_BYTES(x) CUBE_bytes += x;

#define CUBE_END				\
  atomicAdd(CUBE_total_flops, CUBE_flops);	\
  atomicAdd(CUBE_total_bytes, CUBE_bytes);

#define CUBE_INIT() CUBE_Init()
#define CUBE_WRITE() CUBE_Write_Flops()

#elif (CUBE_MODE == CUBE_TIME) // Measure time and calculate Gflop/s and GiByte/s

#define CUBE_KERNEL_CALL CUBE_KERNEL_CALL_TIME
#define CUBE_KERNEL CUBE_KERNEL_DEFAULT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_DEFAULT
#define CUBE_DEVICE CUBE_DEVICE_DEFAULT

#define CUBE_START
#define CUBE_ADD_FLOPS(x)
#define CUBE_ADD_BYTES(x)
#define CUBE_END

#define CUBE_INIT() CUBE_Init()
#define CUBE_WRITE() CUBE_Write_Benchmark()

#else // Default behaviour (do nothing)

#define CUBE_KERNEL_CALL CUBE_KERNEL_CALL_DEFAULT
#define CUBE_KERNEL CUBE_KERNEL_DEFAULT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_DEFAULT
#define CUBE_DEVICE CUBE_DEVICE_DEFAULT

#define CUBE_START
#define CUBE_ADD_FLOPS(x)
#define CUBE_ADD_BYTES(x)
#define CUBE_END

#define CUBE_INIT() 
#define CUBE_WRITE() 

#endif // end CUBE_MODE

#endif // CUBE
