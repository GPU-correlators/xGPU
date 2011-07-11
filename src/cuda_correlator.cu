#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex>
#include <limits.h>
#include <omp.h>

/*
  Data ordering for input vectors is (running from slowest to fastest)
  [time][channel][station][polarization][complexity]

  Output matrix has ordering
  [channel][station][station][polarization][polarization][complexity]
*/

#define USE_GPU

// uncomment to use 8-bit fixed point, comment out for 32-bit floating point
//#define FIXED_POINT

// set the data type accordingly
#ifndef FIXED_POINT
typedef std::complex<float> ComplexInput;
#define COMPLEX_INPUT float2
#define SCALE 1.0f // no rescale required for FP32
#else
typedef std::complex<char> ComplexInput;
#define COMPLEX_INPUT char2 
#define SCALE 16129.0f // need to rescale result 
#endif

// size = freq * time * station * pol *sizeof(ComplexInput)
#define GBYTE (1024llu*1024llu*1024llu)

#define NPOL 2
#define NSTATION 256ll
#define SIGNAL_SIZE GBYTE
#define SAMPLES SIGNAL_SIZE / (NSTATION*NPOL*sizeof(ComplexInput))
#define NFREQUENCY 10ll
#define NTIME 1000ll //SAMPLES / NFREQUENCY
#define NBASELINE ((NSTATION+1)*(NSTATION/2))
#define NDIM 2

//#define PIPE_LENGTH 1
//#define NTIME_PIPE NTIME / PIPE_LENGTH

#define NTIME_PIPE 1000
#define PIPE_LENGTH NTIME / NTIME_PIPE

// how many pulsars are we binning for (Not implemented yet)
#define NPULSAR 0

// whether we are writing the matrix back to device memory (used for benchmarking)
int writeMatrix = 1;

typedef std::complex<float> Complex;

Complex convert(const ComplexInput &b) {
  return Complex(real(b), imag(b));
}

// the OpenMP Xengine
#include "omp_xengine.cc"

// the GPU Xengine
#include "cuda_xengine.cu"

#include "cpu_util.cc"

int main(int argc, char** argv) {

  printf("Correlating %llu stations with %llu signals, with %llu channels and integration length %llu\n",
	 NSTATION, SAMPLES, NFREQUENCY, NTIME);

  unsigned long long vecLength = NFREQUENCY * NTIME * NSTATION * NPOL;

  int fullMatLength = NFREQUENCY * NSTATION*NSTATION*NPOL*NPOL;

  // perform host memory allocation
  int packedMatLength = NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL);

  // allocate the GPU X-engine memory
  ComplexInput *array_h = 0; // this is pinned memory
  Complex *cuda_matrix_h = 0;
  xInit(&array_h, &cuda_matrix_h, NSTATION);

  // create an array of complex noise
  random_complex(array_h, vecLength);

  Complex *omp_matrix_h = (Complex *) malloc(packedMatLength*sizeof(Complex));
  printf("Calling CPU X-Engine\n");
  ompXengine(omp_matrix_h, array_h);

  printf("Calling GPU X-Engine\n");
  cudaXengine(cuda_matrix_h, array_h);

  checkResult(cuda_matrix_h, omp_matrix_h);

  Complex *full_matrix_h = (Complex *) malloc(fullMatLength*sizeof(Complex));

  // convert from packed triangular to full matrix
  extractMatrix(full_matrix_h, cuda_matrix_h);

  //free host memory
  free(omp_matrix_h);

  // free gpu memory
  xFree(array_h, cuda_matrix_h);

  free(full_matrix_h);

  return 0;
}
