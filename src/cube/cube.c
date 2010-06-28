#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <time.h>
#include <string.h>

#include "cube.h"

unsigned int CUBE_nKernel;
ulonglong CUBE_Flops[CUBE_KERNEL_MAX];
ulonglong CUBE_Bytes[CUBE_KERNEL_MAX];
float CUBE_Times[CUBE_KERNEL_MAX];
char *CUBE_Names[CUBE_KERNEL_MAX];

unsigned int CUBE_nCall;

void CUBE_Init(void) {
  int i ;
  CUBE_nCall = CUBE_NCALL;
  CUBE_nKernel = 0;
  for (i=0; i<CUBE_KERNEL_MAX; i++) {
    CUBE_Flops[i] = 0;
    CUBE_Bytes[i] = 0;
    CUBE_Times[i] = 0;
  }
}

int CUBE_Get_Index(const char *kernelName) {
  int i;
  for (i=0; i<CUBE_nKernel; i++) {
    if (strcmp(kernelName, CUBE_Names[i])==0) {
      //printf("Returning index %d for %s\n", i, kernelName);
      return i;
    }
  }
  
  CUBE_Names[i] = (char*)kernelName;
  CUBE_nKernel++;

  if (CUBE_nKernel > CUBE_KERNEL_MAX) {
    printf("CUBE error: cannot track %d kernels, try increasing CUBE_KERNEL_MAX\n",
	   CUBE_nKernel);
    exit(-1);
  }

  printf("Returning index %d for %s\n", CUBE_nKernel-1, kernelName);
  return CUBE_nKernel - 1;
}

// Write out the flop and byte counts to file
void CUBE_Write_Flops(void) {

  FILE *file = fopen("cube_flop_count.log", "w");
  int i;

  fprintf(file, "%d\n", CUBE_nKernel); 
  for (i=0; i<CUBE_nKernel; i++) {
    fprintf(file, "%d %llu %llu\n", i, CUBE_Flops[i], CUBE_Bytes[i]);
  }

  fclose(file);

}

// Read in the flop and byte counts and calculate Gflops and GiBytes rates
void CUBE_Write_Benchmark(void) {

  char filename[1024], filename_csv[1024];
  
  // read in flop and byte counts
  FILE *file_count = fopen("cube_flop_count.log", "r");
  if (file_count == NULL) {
    printf("Error, cube_flop_count.log file not found\n");
    exit(-1);
  }

  int i, Nkernel_file;

  fscanf(file_count, "%d\n", &Nkernel_file); 
  if (CUBE_nKernel != Nkernel_file) {
    printf("Error, read in number of kernels %d, do not match expected value %d\n",
	   Nkernel_file, CUBE_nKernel);
  }

  for (i=0; i<CUBE_nKernel; i++) {
    fscanf(file_count, "%d %llu %llu\n", &i, &(CUBE_Flops[i]), &(CUBE_Bytes[i]));
  }
  fclose(file_count);

  // create a unique log file name 
  //uname(&mach_name);
  //gettimeofday(&now,NULL);
  //strftime(strTime,sizeof(strTime),"%Y%m%d_%H%M%S",localtime(&(now.tv_sec)));
  sprintf(filename,"cube_benchmark.log");
  FILE *file = fopen(filename, "w");

  sprintf(filename_csv,"cube_benchmark.csv");
  FILE *filecsv = fopen(filename_csv, "w");

  fprintf(filecsv, "kernel, time/s, Gflops, Gflops/s, GiBytes, GiBytes/s\n");

  float total_seconds=0, total_gflops=0, total_gbytes=0;
  for (i=0; i<CUBE_nKernel; i++) {
    float seconds = CUBE_Times[i] * 1e-3;
    float gflops = CUBE_Flops[i] * 1e-9;
    float gbytes = CUBE_Bytes[i] / (float)(1<<30);

    total_seconds += seconds;
    total_gflops += gflops;
    total_gbytes += gbytes;

    fprintf(file, "%-40s:   time/s = %6.3f, Gflops = %8.3f, Gflops/s = %8.3f, GiBytes = %8.3f, GiBytes/s = %8.3f\n", 
	    CUBE_Names[i], seconds, gflops, gflops / seconds, gbytes, gbytes / seconds);

    fprintf(filecsv, "%-40s, %6.3f, %8.3f, %8.3f, %8.3f, %8.3f\n", 
	    CUBE_Names[i], seconds, gflops, gflops / seconds, gbytes, gbytes / seconds);
  }

  fprintf(file, "\n");
  fprintf(file, "%-40s:   time/s = %6.3f, Gflops = %8.3f, Gflops/s = %8.3f, GiBytes = %8.3f, GiBytes/s = %8.3f\n", 
	    "Total", total_seconds, total_gflops, total_gflops / total_seconds, 
	  total_gbytes, total_gbytes / total_seconds);

  fprintf(filecsv, "\n");
  fprintf(filecsv, "%-40s, %6.3f, %8.3f, %8.3f, %8.3f, %8.3f\n", 
	  "Total", total_seconds, total_gflops, total_gflops / total_seconds, 
	  total_gbytes, total_gbytes / total_seconds);


  fclose(file);
  fclose(filecsv);
}
