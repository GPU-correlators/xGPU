#ifndef CUDA_XENGINE_H
#define CUDA_XENGINE_H

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
#endif

typedef struct {
  ReImInput real;
  ReImInput imag;
} ComplexInput;

typedef struct {
  float real;
  float imag;
} Complex;

#define NPOL 2
#define NSTATION 256ll
#define NBASELINE ((NSTATION+1)*(NSTATION/2))
#define NFREQUENCY 10ll
#define NTIME 1000ll //SAMPLES / NFREQUENCY
#define NTIME_PIPE 100

// how many pulsars are we binning for (Not implemented yet)
#define NPULSAR 0

// Functions in cuda_xengine.cu

void xInit(ComplexInput **array_h, Complex **matrix_h, int Nstat);

void xFree(ComplexInput *array_h, Complex *matrix_h);

void cudaXengine(Complex *matrix_h, ComplexInput *array_h);

#endif // CUDA_XENGINE_H
