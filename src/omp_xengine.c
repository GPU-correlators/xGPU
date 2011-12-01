/*
  Perform the outer product summation (X-engine) 

  This only fills in the lower triangular matrix and maps the 1-d
  baseline index to the (2-d) triangular index from the formula

  row = floor(-0.5 + sqrt(0.25 + 2*k));
  column = k - row*(row+1)/2

  Not meant to be fast, rather to work in a similar fashion as the GPU
  implementation.

*/

#include <math.h>
#include <omp.h>

#include "xgpu.h"
#include "xgpu_info.h"

#define cxmac(acc,z0,z1)                                                         \
do {                                                                             \
  acc.real += (float)z0.real * (float)z1.real + (float)z0.imag * (float)z1.imag; \
  acc.imag += (float)z0.imag * (float)z1.real - (float)z0.real * (float)z1.imag; \
} while (0)

void ompXengine(Complex *matrix_h, ComplexInput *array_h) {
  int num_procs = omp_get_num_procs();
  #pragma omp parallel num_threads(num_procs)
  {
    int i, t;
    #pragma omp for schedule(dynamic)
    for(i=0; i<NFREQUENCY*NBASELINE; i++){
      int f = i/NBASELINE;
      int k = i - f*NBASELINE;
      int station1 = -0.5 + sqrt(0.25 + 2*k);
      int station2 = k - ((station1+1)*station1)/2;
      Complex sumXX; sumXX.real = 0.0; sumXX.imag = 0.0;
      Complex sumXY; sumXY.real = 0.0; sumXY.imag = 0.0;
      Complex sumYX; sumYX.real = 0.0; sumYX.imag = 0.0;
      Complex sumYY; sumYY.real = 0.0; sumYY.imag = 0.0;
      ComplexInput inputRowX, inputRowY, inputColX, inputColY;
      for(t=0; t<NTIME; t++){
	inputRowX = array_h[((t*NFREQUENCY + f)*NSTATION + station1)*NPOL];
	inputRowY = array_h[((t*NFREQUENCY + f)*NSTATION + station1)*NPOL + 1];
	inputColX = array_h[((t*NFREQUENCY + f)*NSTATION + station2)*NPOL];
	inputColY = array_h[((t*NFREQUENCY + f)*NSTATION + station2)*NPOL + 1];
	cxmac(sumXX, inputRowX, inputColX);
	cxmac(sumXY, inputRowX, inputColY);
	cxmac(sumYX, inputRowY, inputColX);
	cxmac(sumYY, inputRowY, inputColY);
      }
    
      matrix_h[4*i    ] = sumXX;
      matrix_h[4*i + 1] = sumXY;
      matrix_h[4*i + 2] = sumYX;
      matrix_h[4*i + 3] = sumYY;
      // fprintf(stdout,"OUTER:%f %f\n",crealf(matrix_h[4*i]), cimag(matrix_h[4*i]));
    } //end parallel for loop
  }  //end parallel segment
}
