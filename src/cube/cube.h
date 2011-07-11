#ifndef CUBE_H
#define CUBE_H

// maximum number of kernels CUBE can track
#define CUBE_KERNEL_MAX 100

//#if (__CUDA_ARCH__ > 110)
#define ulonglong unsigned long long int
//#else 
//#define ulonglong unsigned int
//#endif

#if defined(__cplusplus)
extern "C" {
#endif

  extern unsigned int CUBE_nCall; // number of times to call each kernel

  extern unsigned int CUBE_nKernel;
  extern ulonglong CUBE_Flops[CUBE_KERNEL_MAX];
  extern ulonglong CUBE_Bytes[CUBE_KERNEL_MAX];
  extern float CUBE_Times[CUBE_KERNEL_MAX];
  extern char CUBE_Names[CUBE_KERNEL_MAX][100];
  extern ulonglong CUBE_Calls[CUBE_KERNEL_MAX];

  extern int CUBE_Async_Active[CUBE_KERNEL_MAX];

  void CUBE_Init(void);
  void CUBE_Write_Flops(void);
  void CUBE_Write_Benchmark(void);
  int CUBE_Get_Index(const char *);
  void CUBE_Print_Kernels();

#if defined(__cplusplus)
}
#endif

// All the C-preprocessor "magic" is below this

#define CUBE_COUNT 1000
#define CUBE_TIME  2000
#define CUBE_ASYNC_COUNT 3000
#define CUBE_ASYNC_TIME 4000
#define CUBE_DEFAULT 5000

#if defined(CUBE_COUNT_MODE)
#define CUBE_MODE CUBE_COUNT // enable for intial flops counting
#elif defined(CUBE_TIME_MODE)
#define CUBE_MODE CUBE_TIME  // enable for final benchmark
#elif defined(CUBE_ASYNC_COUNT_MODE)
#define CUBE_MODE CUBE_ASYNC_COUNT  // enable for asynchronous timing
#elif defined(CUBE_ASYNC_TIME_MODE)
#define CUBE_MODE CUBE_ASYNC_TIME  // enable for asynchronous timing
#else
#define CUBE_MODE CUBE_DEFAULT
#endif

#ifndef CUBE_NCALL
#define CUBE_NCALL 1
#endif

#define SET_COPY_NAME(name, direction, size)				\
  if (direction == cudaMemcpyDeviceToHost)				\
    sprintf(name, "%s_%llu", "cudaMemcpyDeviceToHost", size);		\
  else if (direction == cudaMemcpyHostToDevice)				\
    sprintf(name, "%s_%llu", "cudaMemcpyHostToDevice", size);		\
  else if (direction == cudaMemcpyDeviceToDevice)			\
    sprintf(name, "%s_%llu", "cudaMemcpyDeviceToDevice", size);		\
  else									\
    sprintf(name, "%s_%llu", "cudaMemcpyHostToHost", size);		\

// Counting macro definitions

#define CUBE_ASYNC_KERNEL_CALL_COUNT(kernel, grid, threads, shared, stream, ...) \
  {									\
    ulonglong *d_CUBE_Flops = 0;					\
    ulonglong *d_CUBE_Bytes = 0;					\
    cudaMalloc( (void**)&d_CUBE_Flops, sizeof(ulonglong));		\
    cudaMemset( d_CUBE_Flops, 0, sizeof(ulonglong));			\
    cudaMalloc( (void**)&d_CUBE_Bytes, sizeof(ulonglong)) ;		\
    cudaMemset( d_CUBE_Bytes, 0, sizeof(ulonglong)) ;			\
    kernel <<< grid , threads, shared, stream >>> ( __VA_ARGS__ , d_CUBE_Flops, d_CUBE_Bytes); \
    ulonglong h_CUBE_Flops, h_CUBE_Bytes;				\
    cudaMemcpy(&h_CUBE_Flops, d_CUBE_Flops, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaMemcpy(&h_CUBE_Bytes, d_CUBE_Bytes, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaFree( d_CUBE_Flops );						\
    cudaFree( d_CUBE_Bytes );						\
    int index = CUBE_Get_Index(#kernel);				\
    CUBE_Flops[index] += h_CUBE_Flops;					\
    CUBE_Bytes[index] += h_CUBE_Bytes;					\
    CUBE_Calls[index]++;						\
  }

#define CUBE_KERNEL_COUNT(kernel, ...)					\
  __global__ void kernel(__VA_ARGS__, ulonglong *CUBE_total_flops, ulonglong *CUBE_total_bytes)

#define CUBE_DEVICE_CALL_COUNT(function, ...)	\
  function(__VA_ARGS__, CUBE_flops, CUBE_bytes)

#define CUBE_DEVICE_COUNT(rtn_type, function, ...)			\
  __device__ rtn_type function(__VA_ARGS__, ulonglong &CUBE_flops, ulonglong &CUBE_bytes)

// log the size and direction
#define CUBE_COPY_CALL_COUNT(dst, src, size, direction)			\
  {									\
    cudaMemcpy(dst, src, size, direction);				\
    char copyName[100];							\
    SET_COPY_NAME(copyName, direction, size);				\
    int index = CUBE_Get_Index(copyName);				\
    CUBE_Bytes[index] += size;						\
    CUBE_Calls[index]++;						\
  }

#define CUBE_ASYNC_COPY_CALL_COUNT(dst, src, size, direction, stream)	\
  {									\
    cudaMemcpyAsync(dst, src, size, direction, stream);			\
    char copyName[100];							\
    SET_COPY_NAME(copyName, direction, size);				\
    int index = CUBE_Get_Index(copyName);				\
    CUBE_Bytes[index] += size;						\
    CUBE_Calls[index]++;						\
  }


// Timing macro definitions

#define CUBE_ASYNC_KERNEL_CALL_TIME(kernel, grid, threads, shared, stream, ...) \
  {									\
    kernel <<< grid , threads, shared, stream >>> ( __VA_ARGS__ );	\
    cudaEvent_t start, end;						\
    cudaEventCreate(&start);						\
    cudaEventCreate(&end);						\
    cudaEventSynchronize(start);					\
    cudaEventRecord(start, stream);					\
    for (int i=0; i<CUBE_nCall; i++)					\
      kernel <<< grid, threads, shared, stream >>> ( __VA_ARGS__ );	\
    cudaEventRecord(end, stream);					\
    cudaEventSynchronize(end);						\
    float runTime;							\
    cudaEventElapsedTime(&runTime, start, end);				\
    cudaEventDestroy(start);						\
    cudaEventDestroy(end);						\
    int index = CUBE_Get_Index(#kernel);				\
    CUBE_Times[index] += runTime / CUBE_nCall;				\
  }
  
// log the time based on size and direction
#define CUBE_COPY_CALL_TIME(dst, src, size, direction)			\
  {									\
    cudaEvent_t start, end;						\
    cudaEventCreate(&start);						\
    cudaEventCreate(&end);						\
    cudaEventSynchronize(start);					\
    cudaEventRecord(start, 0);						\
    for (int i=0; i<CUBE_nCall; i++)					\
      cudaMemcpy(dst, src, size, direction);				\
    cudaEventRecord(end, 0);						\
    cudaEventSynchronize(end);						\
    float runTime;							\
    cudaEventElapsedTime(&runTime, start, end);				\
    cudaEventDestroy(start);						\
    cudaEventDestroy(end);						\
    char copyName[100];							\
    SET_COPY_NAME(copyName, direction, size);				\
    int index = CUBE_Get_Index(copyName);				\
    CUBE_Times[index] += runTime / CUBE_nCall;				\
  }

#define CUBE_ASYNC_COPY_CALL_TIME(dst, src, size, direction, stream)	\
  {									\
    cudaEvent_t start, end;						\
    cudaEventCreate(&start);						\
    cudaEventCreate(&end);						\
    cudaEventSynchronize(start);					\
    cudaEventRecord(start, stream);					\
    for (int i=0; i<CUBE_nCall; i++)					\
      cudaMemcpyAsync(dst, src, size, direction, stream);		\
    cudaEventRecord(end, stream);					\
    cudaEventSynchronize(end);						\
    float runTime;							\
    cudaEventElapsedTime(&runTime, start, end);				\
    cudaEventDestroy(start);						\
    cudaEventDestroy(end);						\
    char copyName[100];							\
    SET_COPY_NAME(copyName, direction, size);				\
    int index = CUBE_Get_Index(copyName);				\
    CUBE_Times[index] += runTime / CUBE_nCall;				\
  }


// Async counting definitions

#define CUBE_ASYNC_START_COUNT(label)					\
  {									\
  int index##label = CUBE_Get_Index(#label);				\
  if (CUBE_Async_Active[index##label]) {				\
    printf("Error, %s already active\n", #label);			\
    exit(-1);								\
  }									\
  CUBE_Async_Active[index##label] = 1;					\
  CUBE_Calls[index##label]++;							
  
#define CUBE_ASYNC_END_COUNT(label)					\
  if (!CUBE_Async_Active[index##label]) {				\
    printf("Error, %s doesn't appear to be active\n", #label);		\
    exit(-1);								\
  }									\
  CUBE_Async_Active[index##label] = 0;					\
  }								

#define CUBE_COPY_CALL_ASYNC_COUNT(dst, src, size, direction)		\
  {									\
    cudaMemcpy(dst, src, size, direction);				\
    for (int i=0; i<CUBE_KERNEL_MAX; i++) {				\
      if(CUBE_Async_Active[i]) {					\
	CUBE_Bytes[i] += size;						\
      }									\
    }									\
  }

#define CUBE_ASYNC_COPY_CALL_ASYNC_COUNT(dst, src, size, direction, stream) \
  {									\
    cudaMemcpyAsync(dst, src, size, direction, stream);			\
    for (int i=0; i<CUBE_KERNEL_MAX; i++) {				\
      if(CUBE_Async_Active[i]) {					\
	CUBE_Bytes[i] += size;						\
      }									\
    }									\
  }

#define CUBE_ASYNC_KERNEL_CALL_ASYNC_COUNT(kernel, grid, threads, shared, stream, ...) \
  {									\
    ulonglong *d_CUBE_Flops = 0;					\
    ulonglong *d_CUBE_Bytes = 0;					\
    cudaMalloc( (void**)&d_CUBE_Flops, sizeof(ulonglong));		\
    cudaMemset( d_CUBE_Flops, 0, sizeof(ulonglong));			\
    cudaMalloc( (void**)&d_CUBE_Bytes, sizeof(ulonglong)) ;		\
    cudaMemset( d_CUBE_Bytes, 0, sizeof(ulonglong)) ;			\
    kernel <<< grid , threads, shared, stream >>> ( __VA_ARGS__ , d_CUBE_Flops, d_CUBE_Bytes); \
    ulonglong h_CUBE_Flops, h_CUBE_Bytes;				\
    cudaMemcpy(&h_CUBE_Flops, d_CUBE_Flops, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaMemcpy(&h_CUBE_Bytes, d_CUBE_Bytes, sizeof(ulonglong), cudaMemcpyDeviceToHost); \
    cudaFree( d_CUBE_Flops );						\
    cudaFree( d_CUBE_Bytes );						\
    for (int i=0; i<CUBE_KERNEL_MAX; i++) {				\
      if(CUBE_Async_Active[i]) {					\
	CUBE_Flops[i] += h_CUBE_Flops;					\
	CUBE_Bytes[i] += h_CUBE_Bytes;					\
      }									\
    }									\
  }

// Async timing defintions

#define CUBE_ASYNC_START_TIME(label)		\
  {						\
    int index##label = CUBE_Get_Index(#label);	\
    cudaEvent_t start##label, end##label;	\
    cudaEventCreate(&start##label);		\
    cudaEventCreate(&end##label);		\
    cudaEventSynchronize(start##label);		\
    cudaEventRecord(start##label, 0);			

#define CUBE_ASYNC_END_TIME(label)					\
    cudaEventRecord(end##label, 0);					\
    cudaEventSynchronize(end##label);					\
    float runTime##label;						\
    cudaEventElapsedTime(&runTime##label, start##label, end##label);	\
    cudaEventDestroy(start##label);					\
    cudaEventDestroy(end##label);					\
    CUBE_Times[index##label] += runTime##label;				\
  }

// Default macro definitions

#define CUBE_ASYNC_KERNEL_CALL_DEFAULT(kernel, grid, threads, shared, stream, ...) \
  kernel <<< grid , threads, shared, stream >>>(__VA_ARGS__);

#define CUBE_KERNEL_DEFAULT(kernel, ...)	\
  __global__ void kernel(__VA_ARGS__)

#define CUBE_DEVICE_CALL_DEFAULT(function, ...)	\
  function(__VA_ARGS__)

#define CUBE_DEVICE_DEFAULT(rtn_type, function, ...)	\
  __device__ rtn_type function(__VA_ARGS__)

// plain cudaMemcpy
#define CUBE_COPY_CALL_DEFAULT(dst, src, size, direction)	\
  cudaMemcpy(dst, src, size, direction);

#define CUBE_ASYNC_COPY_CALL_DEFAULT(dst, src, size, direction, stream)	\
  cudaMemcpyAsync(dst, src, size, direction, stream);

// For non-async kernel calls
#define CUBE_KERNEL_CALL(kernel, grid, threads, shared, ...)	\
  CUBE_ASYNC_KERNEL_CALL(kernel, grid, threads, shared, 0, __VA_ARGS__);

// set macros according to mode

#if (CUBE_MODE == CUBE_COUNT) // Count flops and bytes

#define CUBE_COPY_CALL CUBE_COPY_CALL_COUNT
#define CUBE_ASYNC_COPY_CALL CUBE_ASYNC_COPY_CALL_COUNT

#define CUBE_ASYNC_KERNEL_CALL CUBE_ASYNC_KERNEL_CALL_COUNT
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

#define CUBE_ASYNC_START(label)
#define CUBE_ASYNC_END(label)

#elif (CUBE_MODE == CUBE_TIME) // Measure time and calculate Gflop/s and GiByte/s

#define CUBE_COPY_CALL CUBE_COPY_CALL_TIME
#define CUBE_ASYNC_COPY_CALL CUBE_ASYNC_COPY_CALL_TIME

#define CUBE_ASYNC_KERNEL_CALL CUBE_ASYNC_KERNEL_CALL_TIME
#define CUBE_KERNEL CUBE_KERNEL_DEFAULT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_DEFAULT
#define CUBE_DEVICE CUBE_DEVICE_DEFAULT

#define CUBE_START
#define CUBE_ADD_FLOPS(x)
#define CUBE_ADD_BYTES(x)
#define CUBE_END

#define CUBE_INIT() CUBE_Init()
#define CUBE_WRITE() CUBE_Write_Benchmark()

#define CUBE_ASYNC_START(label)
#define CUBE_ASYNC_END(label)

#elif (CUBE_MODE == CUBE_ASYNC_COUNT) // Count flops and bytes for asynchronous kernels and transfers

#define CUBE_COPY_CALL CUBE_COPY_CALL_ASYNC_COUNT
#define CUBE_ASYNC_COPY_CALL CUBE_ASYNC_COPY_CALL_ASYNC_COUNT

#define CUBE_ASYNC_KERNEL_CALL CUBE_ASYNC_KERNEL_CALL_ASYNC_COUNT
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

#define CUBE_ASYNC_START(label) CUBE_ASYNC_START_COUNT(label)
#define CUBE_ASYNC_END(label) CUBE_ASYNC_END_COUNT(label)

#elif (CUBE_MODE == CUBE_ASYNC_TIME) // Measure total times for asynchronous kernels and transfers

#define CUBE_COPY_CALL CUBE_COPY_CALL_DEFAULT
#define CUBE_ASYNC_COPY_CALL CUBE_ASYNC_COPY_CALL_DEFAULT

#define CUBE_ASYNC_KERNEL_CALL CUBE_ASYNC_KERNEL_CALL_DEFAULT
#define CUBE_KERNEL CUBE_KERNEL_DEFAULT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_DEFAULT
#define CUBE_DEVICE CUBE_DEVICE_DEFAULT

#define CUBE_START
#define CUBE_ADD_FLOPS(x)
#define CUBE_ADD_BYTES(x)
#define CUBE_END

#define CUBE_INIT() CUBE_Init()
#define CUBE_WRITE() CUBE_Write_Benchmark()

#define CUBE_ASYNC_START(label) CUBE_ASYNC_START_TIME(label)
#define CUBE_ASYNC_END(label) CUBE_ASYNC_END_TIME(label)

#else // Default behaviour (do nothing)

#define CUBE_ASYNC_KERNEL_CALL CUBE_ASYNC_KERNEL_CALL_DEFAULT
#define CUBE_KERNEL CUBE_KERNEL_DEFAULT
#define CUBE_DEVICE_CALL CUBE_DEVICE_CALL_DEFAULT
#define CUBE_DEVICE CUBE_DEVICE_DEFAULT

#define CUBE_START
#define CUBE_ADD_FLOPS(x)
#define CUBE_ADD_BYTES(x)
#define CUBE_END

#define CUBE_COPY_CALL CUBE_COPY_CALL_DEFAULT
#define CUBE_ASYNC_COPY_CALL CUBE_ASYNC_COPY_CALL_DEFAULT

#define CUBE_INIT() 
#define CUBE_WRITE()

#define CUBE_ASYNC_START(label) 
#define CUBE_ASYNC_END(label) 

#endif // end CUBE_MODE

#endif // CUBE
