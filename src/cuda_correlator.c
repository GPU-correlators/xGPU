#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#ifdef __MACH__
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
int clock_gettime(int clk_id, struct timespec *t){
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds = ((double)time * (double)timebase.numer)/((double)timebase.denom);
    double seconds = ((double)time * (double)timebase.numer)/((double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return 0;
}
#else
#include <time.h>
#endif

#include "cube/cube.h"
#include "xgpu.h"

/*
  Data ordering for input vectors is (running from slowest to fastest)
  [time][channel][station][polarization][complexity]

  Output matrix has ordering
  [channel][station][station][polarization][polarization][complexity]
*/

int main(int argc, char** argv) {

  int opt;
  int i;
  int device = 0;
  unsigned int seed = 1;
  int count = 1;
  int syncOp = SYNCOP_DUMP;
  int finalSyncOp = SYNCOP_DUMP;
  int verbose = 0;
  int hostAlloc = 0;
  XGPUInfo xgpu_info;
  unsigned int npol, nstation, nfrequency;
  int xgpu_error = 0;
  Complex *omp_matrix_h = NULL;
  struct timespec start, stop;
  double total, per_call, max_bw;
#ifdef RUNTIME_STATS
  struct timespec tic, toc;
#endif

  while ((opt = getopt(argc, argv, "c:d:f:ho:rs:v:")) != -1) {
    switch (opt) {
      case 'c':
        // Set number of time to call xgpuCudaXengine
        count = strtoul(optarg, NULL, 0);
        if(count < 1) {
          fprintf(stderr, "count must be positive\n");
          return 1;
        }
        break;
      case 'd':
        // Set CUDA device number
        device = strtoul(optarg, NULL, 0);
        break;
      case 'f':
        // Set syncOp for final call
        finalSyncOp = strtoul(optarg, NULL, 0);
        break;
      case 'o':
        // Set syncOp
        syncOp = strtoul(optarg, NULL, 0);
        break;
      case 'r':
        // Register host allocated memory
        hostAlloc = 1;
        break;
      case 's':
        // Set seed for random data
        seed = strtoul(optarg, NULL, 0);
        break;
      case 'v':
        // Set verbosity level
        verbose = strtoul(optarg, NULL, 0);
        break;
      default: /* '?' */
        fprintf(stderr,
            "Usage: %s [options]\n"
            "Options:\n"
            "  -c COUNT          How many times to call xgpuCudaXengine [1]\n"
            "  -d DEVNUM         GPU device to use [0]\n"
            "  -f FINAL_SYNCOP   Sync operation for final call [1]\n"
            "  -o SYNCOP         Sync operation for all but final call [1]\n"
            "                    Sync operation values are:\n"
            "                         0 (no sync)\n"
            "                         1 (sync and dump)\n"
            "                         2 (sync host to device transfer)\n"
            "                         3 (sync kernel computations)\n"
            "  -r                Register host allocated memory [false]\n"
            "                    (otherwise use CUDA allocated memory)\n"
            "  -s SEED           Random number seed [1]\n"
            "  -v {0|1|2|3}      Verbosity level (debug only) [0]\n"
            "  -h                Show this message\n",
            argv[0]);
        exit(EXIT_FAILURE);
    }
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
  if(hostAlloc) {
    context.array_len = xgpu_info.vecLength;
    context.matrix_len = xgpu_info.matLength;
    context.array_h = malloc(context.array_len*sizeof(ComplexInput));
    context.matrix_h = malloc(context.matrix_len*sizeof(Complex));
  } else {
    context.array_h = NULL;
    context.matrix_h = NULL;
  }
  xgpu_error = xgpuInit(&context, device);
  if(xgpu_error) {
    fprintf(stderr, "xgpuInit returned error code %d\n", xgpu_error);
    goto cleanup;
  }

#ifndef DP4A
  ComplexInput *array_h = context.array_h; // this is pinned memory
#else
  ComplexInput *array_h = malloc(context.array_len*sizeof(ComplexInput));
#endif

  Complex *cuda_matrix_h = context.matrix_h;

  // create an array of complex noise
  xgpuRandomComplex(array_h, xgpu_info.vecLength);

#ifdef DP4A
  xgpuSwizzleInput(context.array_h, array_h);
#endif

  // ompXengine always uses TRIANGULAR_ORDER
  unsigned int ompMatLength = nfrequency * ((nstation+1)*(nstation/2)*npol*npol);
  omp_matrix_h = (Complex *) malloc(ompMatLength*sizeof(Complex));
  if(!omp_matrix_h) {
    fprintf(stderr, "error allocating output buffer for xgpuOmpXengine\n");
    goto cleanup;
  }

#if (CUBE_MODE == CUBE_DEFAULT && !defined(POWER_LOOP) )
  // Only call CPU X engine if dumping GPU X engine
  printf("Calling CPU X-Engine\n");
  xgpuOmpXengine(omp_matrix_h, array_h);
#endif

#define ELAPSED_MS(start,stop) \
  ((((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))/1e6)

  printf("Calling GPU X-Engine\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(i=0; i<count; i++) {
#ifdef RUNTIME_STATS
    clock_gettime(CLOCK_MONOTONIC, &tic);
#endif
    xgpu_error = xgpuCudaXengine(&context, i==count-1 ? finalSyncOp : syncOp);
#ifdef RUNTIME_STATS
    clock_gettime(CLOCK_MONOTONIC, &toc);
#endif
    if(xgpu_error) {
      fprintf(stderr, "xgpuCudaXengine returned error code %d\n", xgpu_error);
      goto cleanup;
    }
#ifdef RUNTIME_STATS
    fprintf(stderr, "%11.6f  %11.6f ms%s\n",
        ELAPSED_MS(start,tic), ELAPSED_MS(tic,toc),
        i==count-1 ? " final" : "");
#endif
  }
  clock_gettime(CLOCK_MONOTONIC, &stop);
  total = ELAPSED_MS(start,stop);
  per_call = total/count;
  // per_spectrum = per_call / NTIME
  // per_channel = per_spectrum / NFREQUENCY
  //             = per_call / (NTIME * NFREQUENCY)
  // max_bw (kHz)  = 1 / per_channel = (NTIME * NFREQUENCY) / per_call
  max_bw = xgpu_info.ntime*xgpu_info.nfrequency/per_call/1000; // MHz
  printf("Elapsed time %.6f ms total, %.6f ms/call average, theoretical max BW %.3f MHz\n",
      total, per_call, max_bw);

#if (CUBE_MODE == CUBE_DEFAULT)
  
  if(count > 1) {
    for(i=0; i<context.matrix_len; i++) {
      cuda_matrix_h[i].real /= count;
      cuda_matrix_h[i].imag /= count;
    }
  }
  xgpuReorderMatrix(cuda_matrix_h);
  xgpuCheckResult(cuda_matrix_h, omp_matrix_h, verbose, array_h);

#if 0
  int fullMatLength = nfrequency * nstation*nstation*npol*npol;
  Complex *full_matrix_h = (Complex *) malloc(fullMatLength*sizeof(Complex));

  // convert from packed triangular to full matrix
  xgpuExtractMatrix(full_matrix_h, cuda_matrix_h);

  free(full_matrix_h);
#endif
#endif

cleanup:
  //free host memory
  free(omp_matrix_h);

  // free gpu memory
  xgpuFree(&context);

#ifdef DP4A
  free(array_h);
#endif

  if(hostAlloc) {
    free(context.array_h);
    free(context.matrix_h);
  }

  return xgpu_error;
}
