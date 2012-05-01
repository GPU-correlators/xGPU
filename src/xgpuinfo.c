#include <stdio.h>

#include "xgpu.h"

int main(int argc, char** argv)
{

  XGPUInfo xgpu_info;

  xgpuInfo(&xgpu_info);

  printf("xGPU library version: %s\n", xgpuVersionString());
  printf("Number of polarizations: %u\n", xgpu_info.npol);
  printf("Number of stations: %u\n", xgpu_info.nstation);
  printf("Number of baselines: %u\n", xgpu_info.nbaseline);
  printf("Number of frequencies: %u\n", xgpu_info.nfrequency);

  printf("Number of time samples per GPU integration: %u\n",
      xgpu_info.ntime);

  printf("Number of time samples per transfer to GPU: %u\n",
      xgpu_info.ntimepipe);

  printf("Type of ComplexInput components: ");
  switch(xgpu_info.input_type) {
    case XGPU_INT8:    printf("8 bit integers\n"); break;
    case XGPU_FLOAT32: printf("32 bit floats\n"); break;
    case XGPU_INT32:   printf("32 bit integers\n"); break;
    default: printf("<unknown type code: %d>\n", xgpu_info.input_type);
  }

  printf("Number of ComplexInput elements in GPU input vector: %llu\n",
      xgpu_info.vecLength);

  printf("Number of ComplexInput elements per transfer to GPU: %llu\n",
      xgpu_info.vecLengthPipe);

  printf("Number of Complex elements in GPU output vector: %llu\n",
      xgpu_info.matLength);

  printf("Number of Complex elements in reordered output vector: %llu\n",
      xgpu_info.triLength);

  printf("Output matrix order: ");
  switch(xgpu_info.matrix_order) {
    case TRIANGULAR_ORDER:               printf("triangular\n");
                                         break;
    case REAL_IMAG_TRIANGULAR_ORDER:     printf("real imaginary triangular\n");
                                         break;
    case REGISTER_TILE_TRIANGULAR_ORDER: printf("register tile triangular\n");
                                         break;
    default: printf("<unknown order code: %d>\n", xgpu_info.matrix_order);
  }

  printf("Shared atomic transfer size: %lu\n", xgpu_info.shared_atomic_size);

  printf("Complex block size: %lu\n", xgpu_info.complex_block_size);

  return 0;
}
