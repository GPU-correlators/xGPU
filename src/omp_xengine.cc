/*
  Perform the outer product summation (X-engine) 

  This only fills in the lower triangular matrix and maps the 1-d
  baseline index to the (2-d) triangular index from the formula

  row = floor(-0.5 + sqrt(0.25 + 2*k));
  column = k - row*(row+1)/2

  Not meant to be fast, rather to work in a similar fashion as the GPU
  implementation.

*/

#include <complex>
#include <omp.h>

#include "omp_xengine.h"

typedef std::complex<float> cppComplex;

inline cppComplex convert(const ComplexInput &b) {
  return cppComplex(b.real, b.imag);
}

void ompXengine(Complex *matrix_h, ComplexInput *array_h) {
  int num_procs = omp_get_num_procs();
  #pragma omp parallel num_threads(num_procs)
  {
    #pragma omp for schedule(dynamic)
    for(int i=0; i<NFREQUENCY*NBASELINE; i++){
      int f = i/NBASELINE;
      int k = i - f*NBASELINE;
      int station1 = -0.5 + sqrt(0.25 + 2*k);
      int station2 = k - ((station1+1)*station1)/2;
      cppComplex sumXX(0.0);
      cppComplex sumXY(0.0);
      cppComplex sumYX(0.0);
      cppComplex sumYY(0.0);
      cppComplex inputRowX, inputRowY, inputColX, inputColY;
      for(int t=0; t<NTIME; t++){
	inputRowX = convert(array_h[((t*NFREQUENCY + f)*NSTATION + station1)*NPOL]);
	inputRowY = convert(array_h[((t*NFREQUENCY + f)*NSTATION + station1)*NPOL + 1]);
	inputColX = convert(array_h[((t*NFREQUENCY + f)*NSTATION + station2)*NPOL]);
	inputColY = convert(array_h[((t*NFREQUENCY + f)*NSTATION + station2)*NPOL + 1]);
	sumXX += inputRowX * conj(inputColX);
	sumXY += inputRowX * conj(inputColY);
	sumYX += inputRowY * conj(inputColX);
	sumYY += inputRowY * conj(inputColY);
      }
    
      matrix_h[4*i    ].real = real(sumXX);
      matrix_h[4*i + 1].real = real(sumXY);
      matrix_h[4*i + 2].real = real(sumYX);
      matrix_h[4*i + 3].real = real(sumYY);
      matrix_h[4*i    ].imag = imag(sumXX);
      matrix_h[4*i + 1].imag = imag(sumXY);
      matrix_h[4*i + 2].imag = imag(sumYX);
      matrix_h[4*i + 3].imag = imag(sumYY);
      // fprintf(stdout,"OUTER:%f %f\n",crealf(matrix_h[4*i]), cimag(matrix_h[4*i]));
    } //end parallel for loop
  }  //end parallel segment
}
