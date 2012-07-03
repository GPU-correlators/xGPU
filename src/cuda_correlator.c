#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
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
  int i, j;
  int device = 0;
  unsigned int seed = 1;
  int outer_count = 1;
  int count = 1;
  int syncOp = SYNCOP_SYNC_TRANSFER;
  int finalSyncOp = SYNCOP_DUMP;
  int verbose = 0;
  int hostAlloc = 0;
  XGPUInfo xgpu_info;
  unsigned int npol, nstation, nfrequency;
  int xgpu_error = 0;
  Complex *omp_matrix_h = NULL;
  struct timespec outer_start, start, stop, outer_stop;
  double total, per_call, max_bw, gbps;
#ifdef RUNTIME_STATS
  struct timespec tic, toc;
#endif

  while ((opt = getopt(argc, argv, "C:c:d:f:ho:rs:v:")) != -1) {
    switch (opt) {
      case 'c':
        // Set number of time to call xgpuCudaXengine
        count = strtoul(optarg, NULL, 0);
        if(count < 1) {
          fprintf(stderr, "count must be positive\n");
          return 1;
        }
        break;
      case 'C':
        // Set number of time to call xgpuCudaXengine
        outer_count = strtoul(optarg, NULL, 0);
        if(outer_count < 1) {
          fprintf(stderr, "outer count must be positive\n");
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
            "  -c INTEG_CALLS    Calls to xgpuCudaXengine per integration [1]\n"
            "  -C INTEG_COUNT    Number of integrations [1]\n"
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
  ComplexInput *array_h = context.array_h; // this is pinned memory
  Complex *cuda_matrix_h = context.matrix_h;

  // create an array of complex noise
  xgpuRandomComplex(array_h, xgpu_info.vecLength);

  // ompXengine always uses TRIANGULAR_ORDER
  unsigned int ompMatLength = nfrequency * ((nstation+1)*(nstation/2)*npol*npol);
  omp_matrix_h = (Complex *) malloc(ompMatLength*sizeof(Complex));
  if(!omp_matrix_h) {
    fprintf(stderr, "error allocating output buffer for xgpuOmpXengine\n");
    goto cleanup;
  }

#if (CUBE_MODE == CUBE_DEFAULT)
  // Only call CPU X engine if dumping GPU X engine
  if(finalSyncOp == SYNCOP_DUMP) {
    printf("Calling CPU X-Engine\n");
    xgpuOmpXengine(omp_matrix_h, array_h);
  }
#endif

#define ELAPSED_MS(start,stop) \
  ((((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))/1e6)

  printf("Calling GPU X-Engine\n");
  clock_gettime(CLOCK_MONOTONIC, &outer_start);
  for(j=0; j<outer_count; j++) {
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
    gbps = ((float)(8 * context.array_len * sizeof(ComplexInput) * count)) / total / 1e6; // Gbps
    printf("Elapsed time %.6f ms total, %.6f ms/call average\n",
        total, per_call);
    printf("Theoretical BW_max %.3f MHz, throughput %.3f Gbps\n",
        max_bw, gbps);
  }
  if(outer_count > 1) {
    clock_gettime(CLOCK_MONOTONIC, &outer_stop);
    total = ELAPSED_MS(outer_start,outer_stop);
    per_call = total/(count*outer_count);
    // per_spectrum = per_call / NTIME
    // per_channel = per_spectrum / NFREQUENCY
    //             = per_call / (NTIME * NFREQUENCY)
    // max_bw (kHz)  = 1 / per_channel = (NTIME * NFREQUENCY) / per_call
    max_bw = xgpu_info.ntime*xgpu_info.nfrequency/per_call/1000; // MHz
    gbps = ((float)(8 * context.array_len * sizeof(ComplexInput) * count * outer_count)) / total / 1e6; // Gbps
    printf("Elapsed time %.6f ms total, %.6f ms/call average\n",
        total, per_call);
    printf("Theoretical BW_max %.3f MHz, throughput %.3f Gbps\n",
        max_bw, gbps);
  }

#if (CUBE_MODE == CUBE_DEFAULT)
  
  // Only compare CPU and GPU X engines if dumping GPU X engine
  if(finalSyncOp == SYNCOP_DUMP) {
    if(count*outer_count > 1) {
      for(i=0; i<context.matrix_len; i++) {
        cuda_matrix_h[i].real /= count*outer_count;
        cuda_matrix_h[i].imag /= count*outer_count;
      }
    }
    xgpuReorderMatrix(cuda_matrix_h);
    xgpuCheckResult(cuda_matrix_h, omp_matrix_h, verbose, array_h);
  }

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

  if(hostAlloc) {
    free(context.array_h);
    free(context.matrix_h);
  }

  return xgpu_error;
}
