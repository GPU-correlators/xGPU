#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <limits.h>

#include "cuda_xengine.h"
#include "omp_xengine.h"
#include "cpu_util.h"

/*
  Data ordering for input vectors is (running from slowest to fastest)
  [time][channel][station][polarization][complexity]

  Output matrix has ordering
  [channel][station][station][polarization][polarization][complexity]
*/

#define USE_GPU

#define TRIANGULAR_ORDER 1000
#define REAL_IMAG_TRIANGULAR_ORDER 2000
#define REGISTER_TILE_TRIANGULAR_ORDER 3000
#define MATRIX_ORDER REGISTER_TILE_TRIANGULAR_ORDER

// size = freq * time * station * pol *sizeof(ComplexInput)
#define GBYTE (1024llu*1024llu*1024llu)

#define SIGNAL_SIZE GBYTE
#define SAMPLES SIGNAL_SIZE / (NSTATION*NPOL*sizeof(ComplexInput))
#define NDIM 2

int main(int argc, char** argv) {

  unsigned int seed = 1;
  int verbose = 0;

  if(argc>1) {
    seed = strtoul(argv[1], NULL, 0);
  }
  if(argc>2) {
    verbose = strtoul(argv[2], NULL, 0);
  }

  srand(seed);

  printf("Correlating %llu stations with %llu signals, with %llu channels and integration length %llu\n",
	 NSTATION, SAMPLES, NFREQUENCY, NTIME);
#ifndef FIXED_POINT
  printf("Sending floating point data to GPU.\n");
#else
  printf("Sending fixed point data to GPU.\n");
#endif

  unsigned long long vecLength = NFREQUENCY * NTIME * NSTATION * NPOL;


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
#if (CUBE_MODE == CUBE_DEFAULT)
  ompXengine(omp_matrix_h, array_h);
#endif

  printf("Calling GPU X-Engine\n");
  cudaXengine(cuda_matrix_h, array_h);

#if (CUBE_MODE == CUBE_DEFAULT)
  
  reorderMatrix(cuda_matrix_h);
  checkResult(cuda_matrix_h, omp_matrix_h, verbose, array_h);

  int fullMatLength = NFREQUENCY * NSTATION*NSTATION*NPOL*NPOL;
  Complex *full_matrix_h = (Complex *) malloc(fullMatLength*sizeof(Complex));

  // convert from packed triangular to full matrix
  extractMatrix(full_matrix_h, cuda_matrix_h);

  free(full_matrix_h);
#endif

  //free host memory
  free(omp_matrix_h);

  // free gpu memory
  xFree(array_h, cuda_matrix_h);

  return 0;
}
