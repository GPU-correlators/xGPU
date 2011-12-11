#ifndef XGPU_H
#define XGPU_H

#ifdef __cplusplus
extern "C" {
#endif

// If FIXED_POINT is defined, the library was compiled to use 8-bit fixed
// point (i.e. integers), otherwise it was compiled to use 32-bit floating
// point (i.e. floats).
#define FIXED_POINT

// set the data type accordingly
#ifndef FIXED_POINT
typedef float ReImInput;
#else
typedef char ReImInput;
#endif // FIXED_POINT

typedef struct ComplexInputStruct {
  ReImInput real;
  ReImInput imag;
} ComplexInput;

typedef struct ComplexStruct {
  float real;
  float imag;
} Complex;

// Used to indicate the size and type of input data
#define XGPU_INT8    (0)
#define XGPU_FLOAT32 (1)
#define XGPU_INT32   (2)

// Used to indicate matrix ordering
#define TRIANGULAR_ORDER 1000
#define REAL_IMAG_TRIANGULAR_ORDER 2000
#define REGISTER_TILE_TRIANGULAR_ORDER 3000

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
  // Output matrix order
  unsigned int matrix_order;
} XGPUInfo;

typedef struct XGPUContextStruct {
  // memory pointers on host
  ComplexInput *array_h;
  Complex *matrix_h;

  // For internal use only
  void *internal;
} XGPUContext;

// Return values from xgpuCudaXengine()
#define XGPU_OK                          (0)
#define XGPU_OUT_OF_MEMORY               (1)
#define XGPU_CUDA_ERROR                  (2)
#define XGPU_INSUFFICIENT_TEXTURE_MEMORY (3)
#define XGPU_NOT_INITIALIZED             (4)

// Functions in cuda_xengine.cu

// Get pointer to library version string.
//
// The library version string should not be modified or freed!
const char * xgpuVersionString();

// Get compile-time sizing parameters.
//
// The XGPUInfo structure pointed to by pcxs is populated with
// compile-time sizing parameters.
void xgpuInfo(XGPUInfo *pcxs);

// Initialize the XGPU.
//
// In addition to allocating device memory and initializing private internal
// context, this routine calls xgpuSetHostInputBuffer() and
// xgpuSetHostOutputBuffer().  Be sure to set the context's array_h and
// matrix_h fields accordingly.
// registered with the CUDA runtime via CudaHostRegister().
//
// If context->array_h is zero, an array of ComplexInput elements is allocated
// (of the appropriate size) via CudaMallocHost, otherwise context->array_h is
// passed to CudaHostRegister.
//
// If context->matrix_h is zero, an array of Complex elements is allocated (of
// the appropriate size) via CudaMallocHost, otherwise context->matrix_h is
// passed to CudaHostRegister.
int xgpuInit(XGPUContext *context);

// Clear the device integration buffer
//
// Sets the device integration buffer to all zeros, effectively starting a new
// integration.
int xgpuClearDeviceIntegrationBuffer(XGPUContext *context);

// Specify a new host input buffer.
//
// The previous host input buffer is freed or unregistered (as required) and
// the value in context->array_h is used to specify the new one.  If
// context->array_h is zero, an array of ComplexInput elements is allocated (of
// the appropriate size) via CudaMallocHost, otherwise cudaHostRegister is
// called to register the region of memory pointed to by context->array_h.  The
// region of memory registered with cudaHostRegister starts at context->array_h
// rounded down to the nearest PAGE_SIZE boundary and has a size equal to the
// vecLength value returned by xgpuInfo() plus the amount, if any, of rounding
// down of context->array_h all rounded up to the next multiple of PAGE_SIZE.
int xgpuSetHostInputBuffer(XGPUContext *context);

// Specify a new host output buffer.
//
// The previous host output buffer is freed or unregistered (as required) and
// the value in context->matrix_h is used to specify the new one.  If
// context->matrix_h is zero, an array of Complex elements is allocated (of the
// appropriate size) via CudaMallocHost, otherwise cudaHostRegister is called
// to register the region of memory pointed to by context->matrix_h.  The
// region of memory registered with cudaHostRegister starts at
// context->matrix_h rounded down to the nearest PAGE_SIZE boundary and has a
// size equal to the matLength value returned by xgpuInfo() plus the amount, if
// any, of rounding down of context->matrix_h all rounded up to the next
// multiple of PAGE_SIZE.
int xgpuSetHostOutputBuffer(XGPUContext *context);

void xgpuFree(XGPUContext *context);

// Perform correlation.  If doDump is non-zero, copy output data back to host.
int xgpuCudaXengine(XGPUContext *context, int doDump);

// Functions in cpu_util.cc

void xgpuRandomComplex(ComplexInput* random_num, long long unsigned int length);

void xgpuReorderMatrix(Complex *matrix);

void xgpuCheckResult(Complex *gpu, Complex *cpu, int verbose, ComplexInput *array_h);

void xgpuExtractMatrix(Complex *matrix, Complex *packed);

// Functions in omp_util.cc

void xgpuOmpXengine(Complex *matrix_h, ComplexInput *array_h);

#ifdef __cplusplus
}
#endif

#endif // XGPU_H
