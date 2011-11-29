#ifndef XGPU_H
#define XGPU_H

#ifdef __cplusplus
extern "C" {
#endif

// Sizing parameters (fixed for now)
#define NPOL 2
#define NSTATION 256ll
#define NBASELINE ((NSTATION+1)*(NSTATION/2))
#define NFREQUENCY 10ll
#define NTIME 1000ll //SAMPLES / NFREQUENCY
#define NTIME_PIPE 100

// how many pulsars are we binning for (Not implemented yet)
#define NPULSAR 0

// uncomment to use 8-bit fixed point, comment out for 32-bit floating point
#define FIXED_POINT

// set the data type accordingly
#ifndef FIXED_POINT
typedef float ReImInput;
#define COMPLEX_INPUT float2
#define SCALE 1.0f // no rescale required for FP32
#else
typedef char ReImInput;
#define COMPLEX_INPUT char2 
#define SCALE 16129.0f // need to rescale result 
#endif // FIXED_POINT

typedef struct ComplexInputStruct {
  ReImInput real;
  ReImInput imag;
} ComplexInput;

typedef struct ComplexStruct {
  float real;
  float imag;
} Complex;

#define XGPU_INT8    (0)
#define XGPU_FLOAT32 (1)
#define XGPU_INT32   (2)

// XGPUInfo is used to convey the compile-time X engine sizing
// parameters of the XGPU library.  It should be allocated by the caller and
// passed (via a pointer) to xInfo() which will fill in the fields.  Note that
// the input values of these fields are currently ignored completely by xInfo()
// (i.e. they are informational only).
typedef struct XGPUInfoStruct {
  // Number of polarizations (NB: will be rolled into a new "ninputs" field)
  unsigned int npol;
  // Number of stations (NB: will be rolled into a new "ninputs" field)
  unsigned int nstation;
  // Number of baselines (derived from nstation)
  unsigned int nbaseline;
  // Number of frequencies
  unsigned int nfrequency;
  // Number of per-channel time samples per integration
  unsigned int ntime;
  // Number of per-channel time samples per transfer to GPU
  unsigned int ntimepipe;
  // Type of input.  One of XGPU_INT8, XGPU_FLOAT32, XGPU_INT32.
  unsigned int input_type;
  // Number of ComplexInput elements in input vector
  long long unsigned int vecLength;
  // Number of ComplexInput elements per transfer to GPU
  long long unsigned int vecLengthPipe;
  // Number of Complex elements in output vector
  long long unsigned int matLength;
} XGPUInfo;

typedef struct XGPUContextStruct {
  // memory pointers on host
  ComplexInput *array_h;
  Complex *matrix_h;

  // For internal use only
  void *internal;
} XGPUContext;

// Functions in cuda_xengine.cu

// Get compile-time sizing parameters.
//
// The XGPUInfo structure pointed to by pcxs is initalized with
// compile-time sizing parameters.
void xInfo(XGPUInfo *pcxs);

// Initialize the XGPU.
//
// In addition to allocating device memory and initializing private internal
// context, the host memory is either allocated via CudaMallocHost() or
// registered with the CUDA runtime via CudaHostRegister().
//
// If context->array_h is zero, an array of ComplexInput elements is allocated
// (of the appropriate size) via CudaMallocHost, otherwise pcontext->array_h is
// passed to CudaHostRegister.
//
// If context->matrix_h is zero, an array of Complex elements is allocated (of
// the appropriate size) via CudaMallocHost, otherwise context->matrix_h is
// passed to CudaHostRegister.
void xInit(XGPUContext *context);

void xFree(XGPUContext *context);

void cudaXengine(XGPUContext *context);

// Functions in cpu_util.cc

void random_complex(ComplexInput* random_num, int length);

void reorderMatrix(Complex *matrix);

void checkResult(Complex *gpu, Complex *cpu, int verbose, ComplexInput *array_h);

void extractMatrix(Complex *matrix, Complex *packed);

// Functions in omp_util.cc

void ompXengine(Complex *matrix_h, ComplexInput *array_h);

#ifdef __cplusplus
}
#endif

#endif // XGPU_H
