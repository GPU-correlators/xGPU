#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cube.h"

unsigned int CUBE_nKernel;
ulonglong CUBE_Calls[CUBE_KERNEL_MAX];
ulonglong CUBE_Flops[CUBE_KERNEL_MAX];
ulonglong CUBE_Bytes[CUBE_KERNEL_MAX];
float CUBE_Times[CUBE_KERNEL_MAX];
char CUBE_Names[CUBE_KERNEL_MAX][100];
int CUBE_Async_Active[CUBE_KERNEL_MAX];

unsigned int CUBE_nCall;

void CUBE_Init(void) {
  int i ;
  CUBE_nCall = CUBE_NCALL;
  CUBE_nKernel = 0;
  for (i=0; i<CUBE_KERNEL_MAX; i++) {
    CUBE_Flops[i] = 0;
    CUBE_Bytes[i] = 0;
    CUBE_Times[i] = 0;
    CUBE_Calls[i] = 0;

    CUBE_Async_Active[i] = 0;
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
  
  strcpy(CUBE_Names[i], kernelName);
  CUBE_nKernel++;

  if (CUBE_nKernel > CUBE_KERNEL_MAX) {
    printf("CUBE error: cannot track %d kernels, try increasing CUBE_KERNEL_MAX\n",
	   CUBE_nKernel);
    exit(-1);
  }

  printf("Returning index %d for %s\n", CUBE_nKernel-1, kernelName);
  return CUBE_nKernel - 1;
}

void CUBE_Print_Kernels(void) {
  int i;
  printf("\nKernel List\n");
  for (i=0; i<CUBE_nKernel; i++) {
    printf("%d %s\n", i, CUBE_Names[i]);
  }
  printf("\n");
}

// Write out the flop and byte counts to file
void CUBE_Write_Flops(void) {

#if (CUBE_MODE == CUBE_COUNT)
  FILE *file = fopen("cube_count.log", "w");
#else
  FILE *file = fopen("cube_async_count.log", "w");
#endif
  int i;

  fprintf(file, "%d\n", CUBE_nKernel); 
  for (i=0; i<CUBE_nKernel; i++) {
    fprintf(file, "%d %llu %llu %llu\n", i, (unsigned long long)CUBE_Calls[i], 
	    (unsigned long long)CUBE_Flops[i], (unsigned long long)CUBE_Bytes[i]);
  }

  fclose(file);

}

// Read in the flop and byte counts and calculate Gflops and GiBytes rates
void CUBE_Write_Benchmark(void) {

  // read in flop and byte counts
#if (CUBE_MODE == CUBE_TIME)
  char *count_file = "cube_count.log";
  char *filename = "cube_benchmark.log";
  char *filename_csv = "cube_benchmark.csv";
#else
  char *count_file = "cube_async_count.log";
  char *filename = "cube_async_benchmark.log";
  char *filename_csv = "cube_async_benchmark.csv";
#endif

  FILE *file_count = fopen(count_file, "r");
  FILE *file = fopen(filename, "w");
  FILE *filecsv = fopen(filename_csv, "w");

  if (file_count == NULL) {
    printf("Error, %s file not found\n", count_file);
    exit(-1);
  }

  int i, Nkernel_file;

  if(fscanf(file_count, "%d\n", &Nkernel_file)== EOF) {
    printf("Error in file header read\n");
    exit(-1);
  }
  if (CUBE_nKernel != Nkernel_file) {
    printf("Error, read in number of kernels %d, do not match expected value %d\n",
	   Nkernel_file, CUBE_nKernel);
  }

  // use a temporary here so the ull doesn't do funny things
  for (i=0; i<CUBE_nKernel; i++) {
    unsigned long long calls, flops, bytes;
    if (fscanf(file_count, "%d %llu %llu %llu\n", &i, &(calls), &(flops), &(bytes)) == EOF) {
      printf("Error reading in line %d\n", i);
      exit(-1);
    }
    CUBE_Calls[i] = (ulonglong)calls;
    CUBE_Flops[i] = (ulonglong)flops;
    CUBE_Bytes[i] = (ulonglong)bytes;
  }
  fclose(file_count);

  fprintf(file, "===============================\n");
#if (CUBE_MODE == CUBE_TIME)
  fprintf(filecsv, "CUBE Kernel profile\n");
  fprintf(file, "CUBE Kernel profile\n");
#else
  fprintf(filecsv, "CUBE Asynchronous block profile\n");
  fprintf(file, "CUBE Asynchronous block profile\n");
#endif
  fprintf(file, "===============================\n\n");

  fprintf(filecsv, "kernel, calls, time/s, Gflops, Gflops/s, GiBytes, GiBytes/s\n");
  fprintf(file, "%-40s: %8s %9s %8s %8s %8s %8s\n", "kernel", "calls", "time/s", "Gflops", "Gflops/s",
	  "GiBytes", "GiBytes/s");

  float total_seconds=0, total_gflops=0, total_gbytes=0;
  ulonglong total_calls=0;

  for (i=0; i<CUBE_nKernel; i++) {
    float seconds = CUBE_Times[i] * 1e-3;
    float gflops = CUBE_Flops[i] * 1e-9;
    float gbytes = CUBE_Bytes[i] / (float)(1<<30);
    ulonglong calls = CUBE_Calls[i];

    total_seconds += seconds;
    total_gflops += gflops;
    total_gbytes += gbytes;
    total_calls += calls;

    fprintf(file, "%-40s: %8llu %9.6f %8.3f %8.3f %8.3f %8.3f\n", 
	    CUBE_Names[i], (unsigned long long)calls, seconds, gflops, gflops / seconds, gbytes, gbytes / seconds);

    fprintf(filecsv, "%-40s, %8llu %9.6f, %8.3f, %8.3f, %8.3f, %8.3f\n", 
	    CUBE_Names[i], (unsigned long long)calls, seconds, gflops, gflops / seconds, gbytes, gbytes / seconds);
  }

  // we don't print totals for the async blocks
#if (CUBE_MODE == CUBE_TIME)
  fprintf(file, "\n");
  fprintf(file, "%-40s: %8llu %9.6f %8.3f %8.3f %8.3f %8.3f\n\n",
	  "Total", (unsigned long long)total_calls, total_seconds, total_gflops, total_gflops / total_seconds, 
	  total_gbytes, total_gbytes / total_seconds);

  fprintf(filecsv, "\n");
  fprintf(filecsv, "%-40s, %8llu %9.6f, %8.3f, %8.3f, %8.3f, %8.3f\n\n", 
	  "Total", (unsigned long long)total_calls, total_seconds, total_gflops, total_gflops / total_seconds, 
	  total_gbytes, total_gbytes / total_seconds);
#endif

  fclose(file);
  fclose(filecsv);
}
