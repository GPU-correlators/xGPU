/*

  Simple cross-product, outputs in correct triangular form.

  - Coalescing memory access in all reads
  - No memory coalscing in writes (will be fixed)
  - Shared memory reads of type float2 to reduce global memory traffic
  - Each thread works on a 2x2 tile of data

  On a GTX 480 with >= 512 tiles this kernel achieve in excess of a
  teraflop.

 */

#include <stdio.h>

#include "xgpu.h"
#include "xgpu_info.h"
#include "xgpu_version.h"
#include "cube/cube.h"

// Set data types accordingly
#ifndef FIXED_POINT
#define COMPLEX_INPUT float2
#define SCALE 1.0f // no rescale required for FP32
#else
#define COMPLEX_INPUT char2 
#define SCALE 16129.0f // need to rescale result 
#endif // FIXED_POINT

// whether we are writing the matrix back to device memory (used for benchmarking)
static int writeMatrix = 1;
// this must be enabled for this option to work though, slightly hurts performance
//#define WRITE_OPTION 

typedef struct XGPUInternalContextStruct {
  //memory pointers on the device
  ComplexInput *array_d[2];
  Complex *matrix_d;

  // used for overlapping comms and compute
  cudaStream_t *streams;

  // texture channel descriptor
  cudaChannelFormatDesc channelDesc;

  // Host input array that we allocated and should free
  ComplexInput * free_array_h;

  // Host input array that we registered and should unregister
  ComplexInput * unregister_array_h;

  // Host output array that we allocated and should free
  Complex * free_matrix_h;

  // Host output array that we registered and should unregister
  Complex * unregister_matrix_h;
} XGPUInternalContext;

#define TILE_HEIGHT 8
#define TILE_WIDTH 8
#define NPOL 2

#define REG_TILE_NBASELINE ((NSTATION/2+1)*(NSTATION/4))

#ifndef FIXED_POINT
// texture declaration for FP32 reads
static texture<float2, 1, cudaReadModeElementType> tex1dfloat2;
static texture<float2, 2, cudaReadModeElementType> tex2dfloat2;
#else
// texture declaration for 8-bit fixed point reads
static texture<char2, 1, cudaReadModeNormalizedFloat> tex1dfloat2;
static texture<char2, 2, cudaReadModeNormalizedFloat> tex2dfloat2;
#endif

// array holding indices for which matrix we are doing the output to at a given iteration
#if (NPULSAR > 0)
static __device__ __constant__ unsigned char tIndex[PIPE_LENGTH*NFREQUENCY];
#endif

#define checkCudaError() do {                           \
    cudaError_t error = cudaGetLastError();		\
    if (error != cudaSuccess) {				\
      fprintf(stderr, "(CUDA) %s", cudaGetErrorString(error));	\
      fprintf(stderr, " (" __FILE__ ":%d)\n", __LINE__);		\
      return XGPU_CUDA_ERROR;						\
    }							\
  } while (0)


//determine row and column from blockIdx.x
CUBE_DEVICE(static void, findPosition, unsigned int &Col, unsigned int &Row, unsigned int &blockX, unsigned int &blockY) {
  unsigned int k = blockIdx.x;
  blockY = -0.5f + sqrtf(0.25f + 2*k);
  blockX = k - (((blockY+1)*(blockY)) >> 1);
  Row = (blockY*TILE_HEIGHT + threadIdx.y);
  Col = (blockX*TILE_WIDTH + threadIdx.x);
}

__device__ static void operator+=( float4 &a, const float4 b ) {
 float4 t = a;
 t.x += b.x; t.y += b.y; t.z += b.z; t.w += b.w;
 a = t;
}

// device function to write out the matrix elements
CUBE_DEVICE(static void, write2x2, unsigned int &Col, unsigned int &Row, float4 *matrix_real, float4 *matrix_imag, 
	    float sum11XXreal, float sum11XXimag, float sum11XYreal, float sum11XYimag,
	    float sum11YXreal, float sum11YXimag, float sum11YYreal, float sum11YYimag,
	    float sum12XXreal, float sum12XXimag, float sum12XYreal, float sum12XYimag,
	    float sum12YXreal, float sum12YXimag, float sum12YYreal, float sum12YYimag,
	    float sum21XXreal, float sum21XXimag, float sum21XYreal, float sum21XYimag,
	    float sum21YXreal, float sum21YXimag, float sum21YYreal, float sum21YYimag,
	    float sum22XXreal, float sum22XXimag, float sum22XYreal, float sum22XYimag,
	    float sum22YXreal, float sum22YXimag, float sum22YYreal, float sum22YYimag) {
  
  int f=blockIdx.y;

#if (MATRIX_ORDER == REGISTER_TILE_TRIANGULAR_ORDER) // write out the register tiles separately
  matrix_real[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*0 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum11XXreal, SCALE*sum11XYreal, SCALE*sum11YXreal, SCALE*sum11YYreal);
  matrix_imag[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*0 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum11XXimag, SCALE*sum11XYimag, SCALE*sum11YXimag, SCALE*sum11YYimag);

  matrix_real[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*1 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum21XXreal, SCALE*sum21XYreal, SCALE*sum21YXreal, SCALE*sum21YYreal);
  matrix_imag[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*1 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum21XXimag, SCALE*sum21XYimag, SCALE*sum21YXimag, SCALE*sum21YYimag);

  matrix_real[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*3 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum22XXreal, SCALE*sum22XYreal, SCALE*sum22YXreal, SCALE*sum22YYreal);
  matrix_imag[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*3 + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum22XXimag, SCALE*sum22XYimag, SCALE*sum22YXimag, SCALE*sum22YYimag);
  
  // Test if entire tile needs to be written or just 3 of 4 parts (exclude top-right)
  if (Col<Row) {
    matrix_real[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*2 + (Row*(Row+1)/2) + Col] += 
      make_float4(SCALE*sum12XXreal, SCALE*sum12XYreal, SCALE*sum12YXreal, SCALE*sum12YYreal);
    matrix_imag[f*4*REG_TILE_NBASELINE + REG_TILE_NBASELINE*2 + (Row*(Row+1)/2) + Col] += 
      make_float4(SCALE*sum12XXimag, SCALE*sum12XYimag, SCALE*sum12YXimag, SCALE*sum12YYimag);
  }
#elif (MATRIX_ORDER == REAL_IMAG_TRIANGULAR_ORDER) // write out the real and imaginary components separately
  Col*=2; Row*=2;
  matrix_real[f*NBASELINE + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum11XXreal, SCALE*sum11XYreal, SCALE*sum11YXreal, SCALE*sum11YYreal);
  matrix_imag[f*NBASELINE + (Row*(Row+1)/2) + Col] += 
    make_float4(SCALE*sum11XXimag, SCALE*sum11XYimag, SCALE*sum11YXimag, SCALE*sum11YYimag);

  matrix_real[f*NBASELINE + ((Row+1)*(Row+2)/2) + Col] += 
    make_float4(SCALE*sum21XXreal, SCALE*sum21XYreal, SCALE*sum21YXreal, SCALE*sum21YYreal);
  matrix_imag[f*NBASELINE + ((Row+1)*(Row+2)/2) + Col] += 
    make_float4(SCALE*sum21XXimag, SCALE*sum21XYimag, SCALE*sum21YXimag, SCALE*sum21YYimag);

  matrix_real[f*NBASELINE + ((Row+1)*(Row+2)/2) + (Col+1)] += 
    make_float4(SCALE*sum22XXreal, SCALE*sum22XYreal, SCALE*sum22YXreal, SCALE*sum22YYreal);
  matrix_imag[f*NBASELINE + ((Row+1)*(Row+2)/2) + (Col+1)] += 
    make_float4(SCALE*sum22XXimag, SCALE*sum22XYimag, SCALE*sum22YXimag, SCALE*sum22YYimag);
  
  // Test if entire tile needs to be written or just 3 of 4 parts (exclude top-right)
  if (Col<Row) {
    matrix_real[f*NBASELINE + (Row*(Row+1)/2) + (Col+1)] += 
      make_float4(SCALE*sum12XXreal, SCALE*sum12XYreal, SCALE*sum12YXreal, SCALE*sum12YYreal);
    matrix_imag[f*NBASELINE + (Row*(Row+1)/2) + (Col+1)] += 
      make_float4(SCALE*sum12XXimag, SCALE*sum12XYimag, SCALE*sum12YXimag, SCALE*sum12YYimag);
  }
#else  // standard triangular packed order
  Col*=2; Row*=2;
  matrix_real[(f*NBASELINE + (Row*(Row+1)/2) + Col)*NPOL + 0] += 
    make_float4(SCALE*sum11XXreal, SCALE*sum11XXimag, SCALE*sum11XYreal, SCALE*sum11XYimag);
  matrix_real[(f*NBASELINE + (Row*(Row+1)/2) + Col)*NPOL + 1] += 
    make_float4(SCALE*sum11YXreal, SCALE*sum11YXimag, SCALE*sum11YYreal, SCALE*sum11YYimag);
  matrix_real[(f*NBASELINE + ((Row+1)*(Row+2)/2) + Col)*NPOL + 0] += 
    make_float4(SCALE*sum21XXreal, SCALE*sum21XXimag, SCALE*sum21XYreal, SCALE*sum21XYimag);
  matrix_real[(f*NBASELINE + ((Row+1)*(Row+2)/2) + Col)*NPOL + 1] += 
    make_float4(SCALE*sum21YXreal, SCALE*sum21YXimag, SCALE*sum21YYreal, SCALE*sum21YYimag);
  matrix_real[(f*NBASELINE + ((Row+1)*(Row+2)/2) + (Col+1))*NPOL + 0] += 
    make_float4(SCALE*sum22XXreal, SCALE*sum22XXimag, SCALE*sum22XYreal, SCALE*sum22XYimag);
  matrix_real[(f*NBASELINE + ((Row+1)*(Row+2)/2) + (Col+1))*NPOL + 1] += 
    make_float4(SCALE*sum22YXreal, SCALE*sum22YXimag, SCALE*sum22YYreal, SCALE*sum22YYimag);
  
  // Test if entire tile needs to be written or just 3 of 4 parts (exclude top-right)
  if (Col<Row) {
    matrix_real[(f*NBASELINE + (Row*(Row+1)/2) + (Col+1))*NPOL + 0] += 
      make_float4(SCALE*sum12XXreal, SCALE*sum12XXimag, SCALE*sum12XYreal, SCALE*sum12XYimag);
    matrix_real[(f*NBASELINE + (Row*(Row+1)/2) + (Col+1))*NPOL + 1] += 
      make_float4(SCALE*sum12YXreal, SCALE*sum12YXimag, SCALE*sum12YYreal, SCALE*sum12YYimag);
  }
#endif

}

// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1
// Read in column in first warp as float2, row in second warp (still true for 1D?)
#define LOAD(s, t)							\
  {float2 temp = tex1Dfetch(tex1dfloat2, array_index + (t)*NFREQUENCY*Nstation*NPOL);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}

#elif TEXTURE_DIM == 2
// Read in column in first warp as float2, row in second warp
#define LOAD(s, t)							\
  {float2 temp = tex2D(tex2dfloat2, array_index, t);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}

#else
#error TEXTURE_DIM must be 1 or 2
#endif

// read in shared data as individual floats to avoid bank conflicts

#define TWO_BY_TWO_COMPUTE(s)						\
  {float col1Xreal = input[s][4*tx];					\
  float col1Ximag = input[s][4*tx + 4*TILE_WIDTH];			\
  float col1Yreal = input[s][4*tx + 1];					\
  float col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH];			\
  float col2Xreal = input[s][4*tx + 2];					\
  float col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH];			\
  float col2Yreal = input[s][4*tx + 3];					\
  float col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH];			\
  float row1Xreal = input[s][4*ty + 8*TILE_WIDTH];			\
  float row1Ximag = input[s][4*ty + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][4*ty + 1 + 8*TILE_WIDTH];			\
  float row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][4*ty + 2 + 8*TILE_WIDTH];			\
  float row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][4*ty + 3 + 8*TILE_WIDTH];			\
  float row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  sum11XXreal += row1Xreal * col1Xreal;					\
  sum11XXreal += row1Ximag * col1Ximag;					\
  sum11XXimag += row1Ximag * col1Xreal;					\
  sum11XXimag -= row1Xreal * col1Ximag;					\
  sum11XYreal += row1Xreal * col1Yreal;					\
  sum11XYreal += row1Ximag * col1Yimag;					\
  sum11XYimag += row1Ximag * col1Yreal;					\
  sum11XYimag -= row1Xreal * col1Yimag;					\
  sum11YXreal += row1Yreal * col1Xreal;					\
  sum11YXreal += row1Yimag * col1Ximag;					\
  sum11YXimag += row1Yimag * col1Xreal;					\
  sum11YXimag -= row1Yreal * col1Ximag;					\
  sum11YYreal += row1Yreal * col1Yreal;					\
  sum11YYreal += row1Yimag * col1Yimag;					\
  sum11YYimag += row1Yimag * col1Yreal;					\
  sum11YYimag -= row1Yreal * col1Yimag;					\
  sum12XXreal += row1Xreal * col2Xreal;					\
  sum12XXreal += row1Ximag * col2Ximag;					\
  sum12XXimag += row1Ximag * col2Xreal;					\
  sum12XXimag -= row1Xreal * col2Ximag;					\
  sum12XYreal += row1Xreal * col2Yreal;					\
  sum12XYreal += row1Ximag * col2Yimag;					\
  sum12XYimag += row1Ximag * col2Yreal;					\
  sum12XYimag -= row1Xreal * col2Yimag;					\
  sum12YXreal += row1Yreal * col2Xreal;					\
  sum12YXreal += row1Yimag * col2Ximag;					\
  sum12YXimag += row1Yimag * col2Xreal;					\
  sum12YXimag -= row1Yreal * col2Ximag;					\
  sum12YYreal += row1Yreal * col2Yreal;					\
  sum12YYreal += row1Yimag * col2Yimag;					\
  sum12YYimag += row1Yimag * col2Yreal;					\
  sum12YYimag -= row1Yreal * col2Yimag;					\
  sum21XXreal += row2Xreal * col1Xreal;					\
  sum21XXreal += row2Ximag * col1Ximag;					\
  sum21XXimag += row2Ximag * col1Xreal;					\
  sum21XXimag -= row2Xreal * col1Ximag;					\
  sum21XYreal += row2Xreal * col1Yreal;					\
  sum21XYreal += row2Ximag * col1Yimag;					\
  sum21XYimag += row2Ximag * col1Yreal;					\
  sum21XYimag -= row2Xreal * col1Yimag;					\
  sum21YXreal += row2Yreal * col1Xreal;					\
  sum21YXreal += row2Yimag * col1Ximag;					\
  sum21YXimag += row2Yimag * col1Xreal;					\
  sum21YXimag -= row2Yreal * col1Ximag;					\
  sum21YYreal += row2Yreal * col1Yreal;					\
  sum21YYreal += row2Yimag * col1Yimag;					\
  sum21YYimag += row2Yimag * col1Yreal;					\
  sum21YYimag -= row2Yreal * col1Yimag;					\
  sum22XXreal += row2Xreal * col2Xreal;					\
  sum22XXreal += row2Ximag * col2Ximag;					\
  sum22XXimag += row2Ximag * col2Xreal;					\
  sum22XXimag -= row2Xreal * col2Ximag;					\
  sum22XYreal += row2Xreal * col2Yreal;					\
  sum22XYreal += row2Ximag * col2Yimag;					\
  sum22XYimag += row2Ximag * col2Yreal;					\
  sum22XYimag -= row2Xreal * col2Yimag;					\
  sum22YXreal += row2Yreal * col2Xreal;					\
  sum22YXreal += row2Yimag * col2Ximag;					\
  sum22YXimag += row2Yimag * col2Xreal;					\
  sum22YXimag -= row2Yreal * col2Ximag;					\
  sum22YYreal += row2Yreal * col2Yreal;					\
  sum22YYreal += row2Yimag * col2Yimag;					\
  sum22YYimag += row2Yimag * col2Yreal;					\
  sum22YYimag -= row2Yreal * col2Yimag;}

CUBE_KERNEL(static shared2x2float2, float4 *matrix_real, float4 *matrix_imag, const int Nstation, const int write)
{
  CUBE_START;

  //get local thread ID
  unsigned int ty = threadIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int tid = ty*TILE_WIDTH + tx;

  //set frequency number from blockIdx.y
  unsigned int f = blockIdx.y;

  unsigned int Row, Col, blockX, blockY;
  CUBE_DEVICE_CALL(findPosition, Col, Row, blockX, blockY);

  //declare shared memory for input coalescing
  __shared__ float input[2][16*TILE_WIDTH]; // 4* for float4, 2* for 2x2 tile size

  //instantiate sum variables
  float sum11XXreal = 0.0, sum11XXimag = 0.0;
  float sum11XYreal = 0.0, sum11XYimag = 0.0;
  float sum11YXreal = 0.0, sum11YXimag = 0.0;
  float sum11YYreal = 0.0, sum11YYimag = 0.0;
  float sum12XXreal = 0.0, sum12XXimag = 0.0;
  float sum12XYreal = 0.0, sum12XYimag = 0.0;
  float sum12YXreal = 0.0, sum12YXimag = 0.0;
  float sum12YYreal = 0.0, sum12YYimag = 0.0;
  float sum21XXreal = 0.0, sum21XXimag = 0.0;
  float sum21XYreal = 0.0, sum21XYimag = 0.0;
  float sum21YXreal = 0.0, sum21YXimag = 0.0;
  float sum21YYreal = 0.0, sum21YYimag = 0.0;
  float sum22XXreal = 0.0, sum22XXimag = 0.0;
  float sum22XYreal = 0.0, sum22XYimag = 0.0;
  float sum22YXreal = 0.0, sum22YXimag = 0.0;
  float sum22YYreal = 0.0, sum22YYimag = 0.0;

  float *input0_p = input[0] + tid;
  float *input1_p = input[1] + tid;
  unsigned int array_index = f*Nstation*NPOL + tid;
  //float array_index = f*Nstation*NPOL + tid;
  if (tid < 4*TILE_WIDTH) {
    array_index += 2*blockX*TILE_WIDTH*NPOL;
  } else {
    array_index += 2*blockY*TILE_WIDTH*NPOL - 4*TILE_HEIGHT;    
    input0_p += 4*TILE_WIDTH;
    input1_p += 4*TILE_WIDTH;
  }


  LOAD(0, 0);

#pragma unroll 2
  for(unsigned int t=0; t<NTIME_PIPE-2; t+=2){
    //for(float t=0.0f; t<(float)NTIME_PIPE-2.0f; /*t+=2.0f*/){

    __syncthreads();

    TWO_BY_TWO_COMPUTE(0);

    //t += 1.0f;
    //LOAD(1, t);    
    LOAD(1, t+1);

    __syncthreads();

    TWO_BY_TWO_COMPUTE(1);

    //t += 1.0f;
    //LOAD(0, t);
    LOAD(0, t+2);
  } 

  __syncthreads();  
  TWO_BY_TWO_COMPUTE(0);

  LOAD(1, NTIME_PIPE-1);

  __syncthreads();

  if (Col > Row) return; // writes seem faster when this is pulled up here
  TWO_BY_TWO_COMPUTE(1);

#ifdef WRITE_OPTION
  if (write) {
#endif
    CUBE_DEVICE_CALL(write2x2, Col, Row, matrix_real, matrix_imag,
		     sum11XXreal, sum11XXimag, sum11XYreal, sum11XYimag, 
		     sum11YXreal, sum11YXimag, sum11YYreal, sum11YYimag, 
		     sum12XXreal, sum12XXimag, sum12XYreal, sum12XYimag, 
		     sum12YXreal, sum12YXimag, sum12YYreal, sum12YYimag, 
		     sum21XXreal, sum21XXimag, sum21XYreal, sum21XYimag, 
		     sum21YXreal, sum21YXimag, sum21YYreal, sum21YYimag, 
		     sum22XXreal, sum22XXimag, sum22XYreal, sum22XYimag, 
		     sum22YXreal, sum22YXimag, sum22YYreal, sum22YYimag);

    CUBE_ADD_BYTES(Col < Row ? 256 : 192); // need load and save
#ifdef WRITE_OPTION
  }
#endif

  CUBE_ADD_FLOPS(NTIME_PIPE*(Col < Row ? 128 : 96));

  CUBE_END;
}

#undef LOAD
#undef TWO_BY_TWO_COMPUTE

static XGPUInfo compiletime_info = {
  npol:          NPOL,
  nstation:      NSTATION,
  nbaseline:     NBASELINE,
  nfrequency:    NFREQUENCY,
  ntime:         NTIME,
  ntimepipe:     NTIME_PIPE,
#ifdef FIXED_POINT
  input_type:    XGPU_INT8,
#else
  input_type:    XGPU_FLOAT32,
#endif
  vecLength:     NFREQUENCY * NTIME * NSTATION * NPOL,
  vecLengthPipe: NFREQUENCY * NTIME_PIPE * NSTATION * NPOL,
#if (MATRIX_ORDER == REGISTER_TILE_TRIANGULAR_ORDER)
  matLength:     NFREQUENCY * ((NSTATION/2+1)*(NSTATION/4)*NPOL*NPOL*4) * (NPULSAR + 1),
#else
  // Matrix length is same for REGISTER_TILE_TRIANGULAR_ORDER and TRIANGULAR_ORDER
  matLength:     NFREQUENCY * ((NSTATION+1)*(NSTATION/2)*NPOL*NPOL) * (NPULSAR + 1),
#endif
  matrix_order:  MATRIX_ORDER
};

// This stringification trick is from "info cpp"
#define STRINGIFY1(s) #s
#define STRINGIFY(s) STRINGIFY1(s)
static const char xgpu_version[] = STRINGIFY(XGPU_VERSION);

const char * xgpuVersionString()
{
  return xgpu_version;
}

// Populate XGPUInfo structure with compile-time parameters.
void xgpuInfo(XGPUInfo *pcxs)
{
  pcxs->npol           = compiletime_info.npol;
  pcxs->nstation       = compiletime_info.nstation;
  pcxs->nbaseline      = compiletime_info.nbaseline;
  pcxs->nfrequency     = compiletime_info.nfrequency;
  pcxs->ntime          = compiletime_info.ntime;
  pcxs->ntimepipe      = compiletime_info.ntimepipe;
  pcxs->input_type     = compiletime_info.input_type;
  pcxs->vecLength      = compiletime_info.vecLength;
  pcxs->vecLengthPipe  = compiletime_info.vecLengthPipe;
  pcxs->matLength      = compiletime_info.matLength;
  pcxs->matrix_order   = compiletime_info.matrix_order;
}

// Initialize the XGPU.
int xgpuInit(XGPUContext *context)
{

  CUBE_INIT();

  // Allocate internal context
  XGPUInternalContext *internal = (XGPUInternalContext *)malloc(sizeof(XGPUInternalContext));
  if(!internal) {
    // Uh-oh!
    return XGPU_OUT_OF_MEMORY;
  }
  context->internal = internal;

  long long unsigned int vecLength = compiletime_info.vecLength;
  long long unsigned int vecLengthPipe = compiletime_info.vecLengthPipe;
  long long unsigned int matLength = compiletime_info.matLength;

  //assign the device
  const int device = 0;
  cudaSetDevice(device); // TODO Put device number in XGPU(Internal?)Context
  checkCudaError();

  if(context->array_h) {
    // Register caller-allocated host memory with CUDA.
    // This requires that the caller allocated the memory properly vis-a-vis
    // the requirements of cudaHostRegister!
    cudaHostRegister(context->array_h, compiletime_info.vecLength*sizeof(ComplexInput), 0);
    internal->unregister_array_h = context->array_h;
    internal->free_array_h = NULL;
  } else {
    // allocate host memory
    cudaMallocHost(&(context->array_h), vecLength*sizeof(ComplexInput));
    internal->free_array_h = context->array_h;
    internal->unregister_array_h = NULL;
    checkCudaError();
  }

  if(context->matrix_h) {
    // Register caller-allocated host memory with CUDA.
    // This requires that the caller allocated the memory properly vis-a-vis
    // the requirements of cudaHostRegister!
    cudaHostRegister(context->matrix_h, compiletime_info.vecLength*sizeof(ComplexInput), 0);
    internal->unregister_matrix_h = context->matrix_h;
    internal->free_matrix_h = NULL;
  } else {
    // allocate host memory
    cudaMallocHost(&(context->matrix_h), matLength*sizeof(Complex));
    internal->free_matrix_h = context->matrix_h;
    internal->unregister_matrix_h = NULL;
    checkCudaError();
  }

  //allocate memory on device
  cudaMalloc((void **) &(internal->array_d[0]), vecLengthPipe*sizeof(ComplexInput));
  cudaMalloc((void **) &(internal->array_d[1]), vecLengthPipe*sizeof(ComplexInput));
  cudaMalloc((void **) &(internal->matrix_d), matLength*sizeof(Complex));
  checkCudaError();
  
  //clear out any previous values
  cudaMemset(internal->array_d[0], '\0', vecLengthPipe*sizeof(ComplexInput));
  cudaMemset(internal->array_d[1], '\0', vecLengthPipe*sizeof(ComplexInput));
  cudaMemset(internal->matrix_d, '\0', matLength*sizeof(Complex));
  checkCudaError();

  // create the streams
  internal->streams = (cudaStream_t*) malloc(2*sizeof(cudaStream_t));
  for(int i=0; i<2; i++) cudaStreamCreate(&(internal->streams[i]));
  checkCudaError();

  internal->channelDesc = cudaCreateChannelDesc<COMPLEX_INPUT>();

#if NPULSAR > 0
  unsigned char timeIndex[PIPE_LENGTH*NFREQUENCY];
  for (int tf=0; tf<PIPE_LENGTH*NFREQUENCY; tf++) timeIndex[tf] = 0;
  cudaMemcpyToSymbol(tIndex, timeIndex, PIPE_LENGTH*NFREQUENCY*sizeof(unsigned char), cudaMemcpyHostToDevice);

  checkCudaError();

  // check symbols are copied over
  unsigned char timeIndex2[PIPE_LENGTH*NFREQUENCY];
  cudaMemcpyFromSymbol(timeIndex2[t], tIndex[t], PIPE_LENGTH*NFREQUENCY*sizeof(unsigned char), cudaMemcpyDeviceToHost);  
  for (int tf=0; tf<PIPE_LENGTH*NFREQUENCY; tf++) {
    for (int f=0; f<NFREQUENCY; f++) 
      if (timeIndex[t][f] != timeIndex2[t][f]) 
	fprintf(stderr, "Index copy failed: t = %d, f = %d, original = %d, copy = %d\n", 
	       t, f, timeIndex[t][f], timeIndex2[t][f]);
  }
#endif

  // check whether texture dimensions are ok
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

#if TEXTURE_DIM == 2
  if((NFREQUENCY * NSTATION * NPOL > deviceProp.maxTexture2D[0]) ||
     (NTIME_PIPE > deviceProp.maxTexture2D[1])) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#elif TEXTURE_DIM == 1
#if 0
  // Surprisingly, this appears not to be a problem with 1D textures.  On a
  // GeForce GTX 580 (i.e. Fermi device), deviceQuery returns 65536 as
  // maxTexture1D, yet the default sizes use 10 * 256 * 2 * 100 * 2 == 1024000
  // bytes of 1D texture without any problems.  Perhaps the value of
  // maxTexture1D returned by cudaGetDeviceProperties is wrong?
  if (NFREQUENCY * NSTATION * NPOL * NTIME_PIPE > deviceProp.maxTexture1D) {
    return XGPU_INSUFFICIENT_TEXTURE_MEMORY;
  }
#endif
#endif 

  return XGPU_OK;
}

// Reinitialize the XGPU host buffers.
int xgpuReinit(XGPUContext *context)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }

  if(context->array_h) {
    if(internal->free_array_h) {
      cudaFreeHost(internal->free_array_h);
    }
    if(internal->unregister_array_h) {
      cudaHostUnregister(internal->unregister_array_h);
    }
    // Register caller-allocated host memory with CUDA.
    // This requires that the caller allocated the memory properly vis-a-vis
    // the requirements of cudaHostRegister!
    cudaHostRegister(context->array_h, compiletime_info.vecLength*sizeof(ComplexInput), 0);
    internal->unregister_array_h = context->array_h;
    internal->free_array_h = NULL;
  }

  if(context->matrix_h) {
    if(internal->free_matrix_h) {
      cudaFreeHost(internal->free_matrix_h);
    }
    if(internal->unregister_matrix_h) {
      cudaHostUnregister(internal->unregister_matrix_h);
    }
    // Register caller-allocated host memory with CUDA.
    // This requires that the caller allocated the memory properly vis-a-vis
    // the requirements of cudaHostRegister!
    cudaHostRegister(context->matrix_h, compiletime_info.vecLength*sizeof(ComplexInput), 0);
    internal->unregister_matrix_h = context->matrix_h;
    internal->free_matrix_h = NULL;
  }

  return XGPU_OK;
}

// Free up the memory on the host and device
void xgpuFree(XGPUContext *context)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;

  if(internal) {
    for(int i=0; i<2; i++)
      cudaStreamDestroy(internal->streams[i]);

    if(internal->free_array_h) {
      cudaFreeHost(internal->free_array_h);
      context->array_h = NULL;
    }
    if(internal->unregister_array_h) {
      cudaHostUnregister(internal->unregister_array_h);
      context->array_h = NULL;
    }
    if(internal->free_matrix_h) {
      cudaFreeHost(internal->free_matrix_h);
      context->matrix_h = NULL;
    }
    if(internal->unregister_matrix_h) {
      cudaHostUnregister(internal->unregister_matrix_h);
      context->matrix_h = NULL;
    }

    cudaFree(internal->array_d[1]);
    cudaFree(internal->array_d[0]);
    cudaFree(internal->matrix_d);

    free(internal);
    context->internal = NULL;
  }

  CUBE_WRITE();
}

int xgpuCudaXengine(XGPUContext *context)
{
  XGPUInternalContext *internal = (XGPUInternalContext *)context->internal;
  if(!internal) {
    return XGPU_NOT_INITIALIZED;
  }

  ComplexInput **array_d = internal->array_d;
  cudaStream_t *streams = internal->streams;
  cudaChannelFormatDesc channelDesc = internal->channelDesc;

  // set pointers to the real and imaginary components of the device matrix
  float4 *matrix_real_d = (float4 *)(internal->matrix_d);
  float4 *matrix_imag_d = (float4 *)(internal->matrix_d + compiletime_info.matLength/2);

  int Nblock = compiletime_info.nstation/min(TILE_HEIGHT,TILE_WIDTH);
  ComplexInput *array_load;
  ComplexInput *array_compute; 

  dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,1);
  //allocated exactly as many thread blocks as are needed
  dim3 dimGrid(((Nblock/2+1)*(Nblock/2))/2, compiletime_info.nfrequency);

  // Create events used to record the completion of the device-host transfer and kernels
  cudaEvent_t copyCompletion[2], kernelCompletion[2];
  for (int i=0; i<2; i++) {
    cudaEventCreate(&kernelCompletion[i]);
    cudaEventCreate(&copyCompletion[i]);
  }
  checkCudaError();

  CUBE_ASYNC_START(ENTIRE_PIPELINE);

  // Need to fill pipeline before loop
  long long unsigned int vecLengthPipe = compiletime_info.vecLengthPipe;
  ComplexInput *array_hp = &context->array_h[0*vecLengthPipe];
  CUBE_ASYNC_COPY_CALL(array_d[0], array_hp, vecLengthPipe*sizeof(ComplexInput), cudaMemcpyHostToDevice, streams[0]);
  cudaEventRecord(copyCompletion[0], streams[0]); // record the completion of the h2d transfer
  checkCudaError();

  CUBE_ASYNC_START(PIPELINE_LOOP);

#ifdef POWER_LOOP
  for (int q=0; ; q++) 
#endif
  for (int p=1; p<PIPE_LENGTH; p++) {
    array_compute = array_d[(p+1)%2];
    array_load = array_d[p%2];

    // Kernel Calculation
#if TEXTURE_DIM == 2
    cudaBindTexture2D(0, tex2dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE, 
		      NFREQUENCY*Nstation*NPOL*sizeof(ComplexInput));
#else
    cudaBindTexture(0, tex1dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*NTIME_PIPE*sizeof(ComplexInput));
#endif
    cudaStreamWaitEvent(streams[1], copyCompletion[(p+1)%2], 0); // only start the kernel once the h2d transfer is complete
    CUBE_ASYNC_KERNEL_CALL(shared2x2float2, dimGrid, dimBlock, 0, streams[1], 
			   matrix_real_d, matrix_imag_d, NSTATION, writeMatrix);
    cudaEventRecord(kernelCompletion[(p+1)%2], streams[1]); // record the completion of the h2d transfer
    checkCudaError();

    // Download input data
    ComplexInput *array_hp = &context->array_h[p*vecLengthPipe];
    cudaStreamWaitEvent(streams[0], kernelCompletion[p%2], 0); // only start the transfer once the kernel has completed
    CUBE_ASYNC_COPY_CALL(array_load, array_hp, vecLengthPipe*sizeof(ComplexInput), cudaMemcpyHostToDevice, streams[0]);
    cudaEventRecord(copyCompletion[p%2], streams[0]); // record the completion of the h2d transfer
    checkCudaError();
  }

  CUBE_ASYNC_END(PIPELINE_LOOP);

  array_compute = array_d[(PIPE_LENGTH+1)%2];
  // Final kernel calculation
#if TEXTURE_DIM == 2
  cudaBindTexture2D(0, tex2dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL, NTIME_PIPE, 
		    NFREQUENCY*Nstation*NPOL*sizeof(ComplexInput));
#else
    cudaBindTexture(0, tex1dfloat2, array_compute, channelDesc, NFREQUENCY*NSTATION*NPOL*NTIME_PIPE*sizeof(ComplexInput));
#endif
  cudaStreamWaitEvent(streams[1], copyCompletion[(PIPE_LENGTH+1)%2], 0);
  CUBE_ASYNC_KERNEL_CALL(shared2x2float2, dimGrid, dimBlock, 0, streams[1], matrix_real_d, matrix_imag_d,
			 NSTATION, writeMatrix);
  checkCudaError();

  //copy the data back, employing a similar strategy as above
  CUBE_COPY_CALL(context->matrix_h, internal->matrix_d, compiletime_info.matLength*sizeof(Complex), cudaMemcpyDeviceToHost);
  checkCudaError();

  CUBE_ASYNC_END(ENTIRE_PIPELINE);

  for (int i=0; i<2; i++) {
    cudaEventDestroy(copyCompletion[i]);
    cudaEventDestroy(kernelCompletion[i]);
  }
  checkCudaError();

  return XGPU_OK;
}
