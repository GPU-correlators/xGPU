// random numbers in the range 
void random_complex(ComplexInput* random_num, int length) {
  float a,b;
  for(int i=0; i<length; i++){
    a = ((rand()-RAND_MAX/2) / (float)(RAND_MAX/2));
    b = ((rand()-RAND_MAX/2) / (float)(RAND_MAX/2));
    random_num[i] = ComplexInput(SCALE*a,SCALE*b);
  }
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
