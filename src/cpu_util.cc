// random numbers in the range 
void random_complex(ComplexInput* random_num, int length) {
  float a,b;
  for(int i=0; i<length; i++){
    a = ((rand()-RAND_MAX/2) / (float)(RAND_MAX/2));
    b = ((rand()-RAND_MAX/2) / (float)(RAND_MAX/2));
    random_num[i] = ComplexInput(SCALE*a,SCALE*b);
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
#define TOL 1e-1
void checkResult(Complex *gpu, Complex *cpu) {

  printf("Checking result...\n"); fflush(stdout);

  int error=0;

  for(int f=0; f<NFREQUENCY; f++){
    for(int i=0; i<NSTATION; i++){
      for (int j=0; j<=i; j++) {
	int k = f*(NSTATION+1)*(NSTATION/2) + i*(i+1)/2 + j;
        for (int pol1=0; pol1<NPOL; pol1++) {
	  for (int pol2=0; pol2<NPOL; pol2++) {
	    int index = (k*NPOL+pol1)*NPOL+pol2;
	    if (abs(cpu[index] - gpu[index]) / abs(cpu[index]) > TOL) {
	      printf("%d %d %d %d %d %d %d %f  %f  %f  %f\n", f, i, j, k, pol1, pol2, index, 
		     real(cpu[index]), real(gpu[index]), imag(cpu[index]), imag(gpu[index]));
	      error++;
	    }
	  }
	}
      }
    }
  }

  if (error) {
    printf("Outer product summation failed with %d deviations\n\n", error);    
  } else {
    printf("Outer product summation successful\n\n");
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
