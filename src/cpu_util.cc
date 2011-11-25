#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_xengine.h" // For Complex and CompexInput typedefs

// Normally distributed random numbers with standard deviation of 2.5,
// quantized to integer values and saturated to the range -7.0 to +7.0.  For
// the fixed point case, the values are then converted to ints, scaled by 16
// (i.e. -112 to +112), and finally stored as signed chars.
void random_complex(ComplexInput* random_num, int length) {
  double u1,u2,r,theta,a,b;
  double stddev=2.5;
  for(int i=0; i<length; i++){
    u1 = (rand() / (double)(RAND_MAX));
    u2 = (rand() / (double)(RAND_MAX));
    if(u1==0.0) u1=0.5/RAND_MAX;
    if(u2==0.0) u2=0.5/RAND_MAX;
    // Do Box-Muller transform
    r = stddev * sqrt(-2.0*log(u1));
    theta = 2*M_PI*u2;
    a = r * cos(theta);
    b = r * sin(theta);
    // Quantize (TODO: unbiased rounding?)
    a = round(a);
    b = round(b);
    // Saturate
    if(a >  7.0) a =  7.0;
    if(a < -7.0) a = -7.0;
    if(b >  7.0) b =  7.0;
    if(b < -7.0) b = -7.0;
#ifndef FIXED_POINT
    // Simulate 4 bit data that has been converted to floats
    // (i.e. {-7.0, -6.0, ..., +6.0, +7.0})
    random_num[i] = ComplexInput( a, b );
#else
    // Simulate 4 bit data that has been multipled by 16 (via left shift by 4;
    // could multiply by 18 to maximize range, but that might be more expensive
    // than left shift by 4).
    // (i.e. {-112, -96, -80, ..., +80, +96, +112})
    random_num[i] = ComplexInput( ((int)a) << 4, ((int)b) << 4 );

    // Uncomment next line to simulate all zeros for every input.
    // Interestingly, it does not give exactly zeros on the output.
    //random_num[i] = ComplexInput(0,0);
#endif
  }
}

void reorderMatrix(Complex *matrix) {

#if MATRIX_ORDER == REGISTER_TILE_TRIANGULAR_ORDER
  // reorder the matrix from REGISTER_TILE_TRIANGULAR_ORDER to TRIANGULAR_ORDER

  size_t matLength = NFREQUENCY * ((NSTATION/2+1)*(NSTATION/4)*NPOL*NPOL*4) * (NPULSAR + 1);
  Complex *tmp = new Complex[matLength];
  memset(tmp, '0', matLength);

  for(int f=0; f<NFREQUENCY; f++) {
    for(int i=0; i<NSTATION/2; i++) {
      for (int rx=0; rx<2; rx++) {
	for (int j=0; j<=i; j++) {
	  for (int ry=0; ry<2; ry++) {
	    int k = f*(NSTATION+1)*(NSTATION/2) + (2*i+rx)*(2*i+rx+1)/2 + 2*j+ry;
	    int l = f*4*(NSTATION/2+1)*(NSTATION/4) + (2*ry+rx)*(NSTATION/2+1)*(NSTATION/4) + i*(i+1)/2 + j;
	    for (int pol1=0; pol1<NPOL; pol1++) {
	      for (int pol2=0; pol2<NPOL; pol2++) {
		size_t tri_index = (k*NPOL+pol1)*NPOL+pol2;
		size_t reg_index = (l*NPOL+pol1)*NPOL+pol2;
		tmp[tri_index] = 
		  Complex(((float*)matrix)[reg_index], ((float*)matrix)[reg_index+matLength]);
	      }
	    }
	  }
	}
      }
    }
  }
   
  memcpy(matrix, tmp, matLength*sizeof(Complex));

  delete []tmp;

#elif MATRIX_ORDER == REAL_IMAG_TRIANGULAR_ORDER
  // reorder the matrix from REAL_IMAG_TRIANGULAR_ORDER to TRIANGULAR_ORDER
  
  size_t matLength = NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1);
  Complex *tmp = new Complex[matLength];

  for(int f=0; f<NFREQUENCY; f++){
    for(int i=0; i<NSTATION; i++){
      for (int j=0; j<=i; j++) {
	int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
        for (int pol1=0; pol1<NPOL; pol1++) {
	  for (int pol2=0; pol2<NPOL; pol2++) {
	    size_t index = (k*NPOL+pol1)*NPOL+pol2;
	    tmp[index] = Complex(((float*)matrix)[index], ((float*)matrix)[index+matLength]);
	  }
	}
      }
    }
  }

  memcpy(matrix, tmp, matLength*sizeof(Complex));

  delete []tmp;
#endif

  return;
}

//check that GPU calculation matches the CPU
//
// verbose=0 means just print summary.
// verbsoe=1 means print each differing basline/channel.
// verbose=2 and array_h!=0 means print each differing baseline and each input
//           sample that contributed to it.
#ifndef FIXED_POINT
#define TOL 1e-12
#else
#define TOL 1e-5
#endif // FIXED_POINT
void checkResult(Complex *gpu, Complex *cpu, int verbose=0, ComplexInput *array_h=0) {

  printf("Checking result (tolerance == %g)...\n", TOL); fflush(stdout);

  int errorCount=0;
  double error = 0.0;
  double maxError = 0.0;

  for(int i=0; i<NSTATION; i++){
    for (int j=0; j<=i; j++) {
      for (int pol1=0; pol1<NPOL; pol1++) {
	for (int pol2=0; pol2<NPOL; pol2++) {
	  for(int f=0; f<NFREQUENCY; f++){
	    int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
	    int index = (k*NPOL+pol1)*NPOL+pol2;
	    if(abs(cpu[index]) == 0) {
	      error = abs(gpu[index]);
	    } else {
	      error = abs(cpu[index] - gpu[index]) / abs(cpu[index]);
	    }
	    if(error > maxError) {
	      maxError = error;
	    }
	    if(error > TOL) {
              if(verbose > 0) {
                printf("%d %d %d %d %d %d %d %g  %g  %g  %g (%g %g)\n", f, i, j, k, pol1, pol2, index,
                       real(cpu[index]), real(gpu[index]), imag(cpu[index]), imag(gpu[index]), abs(cpu[index]), abs(gpu[index]));
                if(verbose > 1 && array_h) {
                  Complex sum(0,0);
                  for(int t=0; t<NTIME; t++) {
                    ComplexInput in0 = array_h[t*NFREQUENCY*NSTATION*2 + f*NSTATION*2 + i*2 + pol1];
                    ComplexInput in1 = array_h[t*NFREQUENCY*NSTATION*2 + f*NSTATION*2 + j*2 + pol2];
                    Complex prod = convert(in0) * conj(convert(in1));
                    sum += prod;
                    printf(" %4d (%4g,%4g) (%4g,%4g) -> (%6g, %6g)\n", t,
                        (float)real(in0), (float)imag(in0),
                        (float)real(in1), (float)imag(in1),
                        (float)real(prod), (float)imag(prod));
                  }
                  printf("                                 (%6g, %6g)\n", real(sum), imag(sum));
                }
              }
	      errorCount++;
	    }
	  }
	}
      }
    }
  }

  if (errorCount) {
    printf("Outer product summation failed with %d deviations (max error %g)\n\n", errorCount, maxError);
  } else {
    printf("Outer product summation successful (max error %g)\n\n", maxError);
  }

}

// Extracts the full matrix from the packed Hermitian form
void extractMatrix(Complex *matrix, Complex *packed) {

  for(int f=0; f<NFREQUENCY; f++){
    for(int i=0; i<NSTATION; i++){
      for (int j=0; j<=i; j++) {
	int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
        for (int pol1=0; pol1<NPOL; pol1++) {
	  for (int pol2=0; pol2<NPOL; pol2++) {
	    int index = (k*NPOL+pol1)*NPOL+pol2;
	    matrix[(((f*NSTATION + i)*NSTATION + j)*NPOL + pol1)*NPOL+pol2] = 
	      packed[index];
	    matrix[(((f*NSTATION + j)*NSTATION + i)*NPOL + pol2)*NPOL+pol1] = conj(packed[index]);
	    //printf("%d %d %d %d %d %d %d\n",f,i,j,k,pol1,pol2,index);
	  }
	}
      }
    }
  }

}
