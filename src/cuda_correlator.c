#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>

#include "xgpu.h"

/*
  Data ordering for input vectors is (running from slowest to fastest)
  [time][channel][station][polarization][complexity]

  Output matrix has ordering
  [channel][station][station][polarization][polarization][complexity]
*/

int main(int argc, char** argv) {

  unsigned int seed = 1;
  int verbose = 0;
  XGPUInfo xgpu_info;
  unsigned int npol, nstation, nfrequency;

  if(argc>1) {
    seed = strtoul(argv[1], NULL, 0);
  }
  if(argc>2) {
    verbose = strtoul(argv[2], NULL, 0);
  }

  srand(seed);

  // Get sizing info from library
  xgpuInfo(&xgpu_info);
  npol = xgpu_info.npol;
  nstation = xgpu_info.nstation;
  nfrequency = xgpu_info.nfrequency;

  printf("Correlating %u stations with %u channels and integration length %u\n",
	 xgpu_info.nstation, xgpu_info.nfrequency, xgpu_info.ntime);
#ifndef FIXED_POINT
  printf("Sending floating point data to GPU.\n");
#else
  printf("Sending fixed point data to GPU.\n");
#endif

  // perform host memory allocation

  // allocate the GPU X-engine memory
  XGPUContext context;
  context.array_h = NULL;
  context.matrix_h = NULL;
  xgpuInit(&context);
  ComplexInput *array_h = context.array_h; // this is pinned memory
  Complex *cuda_matrix_h = context.matrix_h;

  // create an array of complex noise
  xgpuRandomComplex(array_h, xgpu_info.vecLength);

  // ompXengine always uses TRIANGULAR_ORDER
  int ompMatLength = nfrequency * ((nstation+1)*(nstation/2)*npol*npol);
  Complex *omp_matrix_h = (Complex *) malloc(ompMatLength*sizeof(Complex));
  printf("Calling CPU X-Engine\n");
#if (CUBE_MODE == CUBE_DEFAULT)
  xgpuOmpXengine(omp_matrix_h, array_h);
#endif

  printf("Calling GPU X-Engine\n");
  xgpuCudaXengine(&context);

#if (CUBE_MODE == CUBE_DEFAULT)
  
  xgpuReorderMatrix(cuda_matrix_h);
  xgpuCheckResult(cuda_matrix_h, omp_matrix_h, verbose, array_h);

  int fullMatLength = nfrequency * nstation*nstation*npol*npol;
  Complex *full_matrix_h = (Complex *) malloc(fullMatLength*sizeof(Complex));

  // convert from packed triangular to full matrix
  xgpuExtractMatrix(full_matrix_h, cuda_matrix_h);

  free(full_matrix_h);
#endif

  //free host memory
  free(omp_matrix_h);

  // free gpu memory
  xgpuFree(&context);

  return 0;
}
