//determine row and column from blockIdx.x
template<bool offdiag>
CUBE_DEVICE(static void, findPosition, unsigned int &Col, unsigned int &Row, unsigned int &blockX, unsigned int &blockY) {
  unsigned int k = blockIdx.x;

  if (!offdiag) { // update all of lower triangular
#if NSTATION >= 512
    blockY = -0.5 + sqrt(0.25 + 2*k);
#else
    blockY = -0.5f + sqrtf(0.25f + 2*k);
#endif  
    blockX = k - (((blockY+1)*(blockY)) >> 1);
  } else { // only update strictly lower block triangular
#if NSTATION >= 512
    blockY = 0.5 + sqrt(0.25 + 2*k);
#else
    blockY = 0.5f + sqrtf(0.25f + 2*k);
#endif  
    blockX = k - (((blockY-1)*(blockY)) >> 1);
  }
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

#ifndef COMPLEX_BLOCK_SIZE
#define COMPLEX_BLOCK_SIZE 1
#endif

#if COMPLEX_BLOCK_SIZE != 1 && COMPLEX_BLOCK_SIZE != 32
#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif

//#define STRUCT_OF_ARRAY

// Use the appropriate shared memory load / store routines according to the atomic size
#if SHARED_ATOMIC_SIZE == 4
#include "shared_transfer_4.cuh"
#elif SHARED_ATOMIC_SIZE == 8
#include "shared_transfer_8.cuh"
#else
#error SHARED_ATOMIC_SIZE must be 4 or 8
#endif

template<bool offdiag>
CUBE_KERNEL(static shared2x2float2, float4 *matrix_real, float4 *matrix_imag, const int Nstation, const int write)
{
  CUBE_START;

// Set the degree of shared memory buffering to use
#if __CUDA_ARCH__ < 300 
#define BUFFER_DEPTH 2 // Fermi optimal setting
#else 
#define BUFFER_DEPTH 4 // Kepler optimal setting
#endif

  //get local thread ID
  unsigned int ty = threadIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int tid = ty*TILE_WIDTH + tx;

  //set frequency number from blockIdx.y
  unsigned int f = blockIdx.y;

  unsigned int Row, Col, blockX, blockY;
  CUBE_DEVICE_CALL(findPosition<offdiag>, Col, Row, blockX, blockY);

  //declare shared memory for input coalescing

#if SHARED_ATOMIC_SIZE == 4
  __shared__ float input[BUFFER_DEPTH][16*TILE_WIDTH]; // 4* for float4, 4* for 2x2 tile size
  float *input0_p = input[0] + tid;
  float *input1_p = input[1] + tid;
#if BUFFER_DEPTH == 4
  float *input2_p = input[2] + tid;
  float *input3_p = input[3] + tid;
#endif // BUFFER_DEPTH==4
#else
  __shared__ float2 input[BUFFER_DEPTH][8*TILE_WIDTH]; // 2* for float4/float2, 4* for 2x2 tile size

#ifdef STRUCT_OF_ARRAY
  unsigned swizzled_tid = ((tid & 0x1c) >> 1) | ((tid & 2)    << 3) | (tid & 0x21);
  float2 *input0_p = input[0] + swizzled_tid;
  float2 *input1_p = input[1] + swizzled_tid;
#if BUFFER_DEPTH == 4
  float2 *input2_p = input[2] + swizzled_tid;
  float2 *input3_p = input[3] + swizzled_tid;
#endif // BUFFER_DEPTH==4

#else

  float2 *input0_p = input[0] + tid;
  float2 *input1_p = input[1] + tid;
#if BUFFER_DEPTH == 4
  float2 *input2_p = input[2] + tid;
  float2 *input3_p = input[3] + tid;
#endif // BUFFER_DEPTH==4
#endif // STRUCT_OF_ARRAY

#endif

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

#if SHARED_ATOMIC_SIZE == 8
#if COMPLEX_BLOCK_SIZE != 1
#error COMPLEX_BLOCK_SIZE must be 1 for SHARED_ATOMIC_SIZE == 8 (for now)
#endif
#endif

  unsigned int array_index = f*Nstation*NPOL + tid;

  if (tid < 4*TILE_WIDTH) {
    // Read in column in first warp
    array_index += 2*blockX*TILE_WIDTH*NPOL;
  } else {
    // Read in row in second warp
    array_index += 2*blockY*TILE_WIDTH*NPOL - 4*TILE_HEIGHT;    
#if SHARED_ATOMIC_SIZE == 4
    // threads 32..63 now have offset 64..95
    input0_p += 4*TILE_WIDTH;
    input1_p += 4*TILE_WIDTH;
#if BUFFER_DEPTH==4
    input2_p += 4*TILE_WIDTH;
    input3_p += 4*TILE_WIDTH;
#endif // BUFFER_DEPTH=4
#endif
  }

#if BUFFER_DEPTH==2
  LOAD(0, 0);
#elif BUFFER_DEPTH==4
  LOAD(0, 0);
  LOAD(1, 1);
#endif

#if __CUDA_ARCH__ < 300
#pragma unroll 2
#else
#pragma unroll 1
#endif
  for(unsigned int t=0; t<NTIME_PIPE-BUFFER_DEPTH; t+=BUFFER_DEPTH){

    __syncthreads();

#if BUFFER_DEPTH==2
    TWO_BY_TWO_COMPUTE(0);
    LOAD(1, t+1);
#elif BUFFER_DEPTH==4
    TWO_BY_TWO_COMPUTE(0);
    TWO_BY_TWO_COMPUTE(1);
    LOAD(2, t+2);
    LOAD(3, t+3);
#endif

    __syncthreads();


#if BUFFER_DEPTH==2
    TWO_BY_TWO_COMPUTE(1);
    LOAD(0, t+2);
#elif BUFFER_DEPTH==4
    TWO_BY_TWO_COMPUTE(2);
    TWO_BY_TWO_COMPUTE(3);
    LOAD(0, t+4);
    LOAD(1, t+5);
#endif

  } 

  __syncthreads();  

#if BUFFER_DEPTH==2
  TWO_BY_TWO_COMPUTE(0);
  LOAD(1, NTIME_PIPE-1);
#elif BUFFER_DEPTH==4
  TWO_BY_TWO_COMPUTE(0);
  TWO_BY_TWO_COMPUTE(1);
  LOAD(2, NTIME_PIPE-2);
  LOAD(3, NTIME_PIPE-1);
#endif

  __syncthreads();

#if BUFFER_DEPTH==2
  TWO_BY_TWO_COMPUTE(1);
#elif BUFFER_DEPTH==4
  TWO_BY_TWO_COMPUTE(2);
  TWO_BY_TWO_COMPUTE(3);
#endif

  if (Col > Row) return; // writes seem faster when this is pulled up here

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
#undef TWO_BY_TWO_PRELOAD
