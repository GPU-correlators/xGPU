// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1

// Read char4 from global, write int to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  { int2 c = tex1Dfetch(tex1dchar4, array_index + (t)*NFREQUENCY*NSTATION*NPOL); \
    CUBE_ADD_BYTES(4*sizeof(ComplexInput));				\
    *(input##s##_p) = c.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = c.y;}

#else

//#define TEXTURE_FLOAT_COORD
#ifndef TEXTURE_FLOAT_COORD

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  {  int4 c;								\
    asm("tex.2d.v4.s32.s32 {%0, %1, %2, %3}, [tex2dchar4, {%4, %5}];" :	\
	"=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "r"(array_index), "r"(t)); \
    CUBE_ADD_BYTES(4*sizeof(ComplexInput));				\
    *(input##s##_p) = c.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = c.y;}

#else

// Read char4 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  { int2 c = tex2D(tex2dchar4, array_index, t);				\
    CUBE_ADD_BYTES(4*sizeof(ComplexInput));				\
    *(input##s##_p) = c.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = c.y;}
#endif  // use float texture coordinates

#endif

// read in shared data as individual floats to avoid bank conflicts

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {int col1Xreal = input[s][4*tx                                   ];	\
  int col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  int col1Yreal = input[s][4*tx + 1                               ];	\
  int col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  int col2Xreal = input[s][4*tx + 2                               ];	\
  int col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  int col2Yreal = input[s][4*tx + 3                               ];	\
  int col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
  int row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  int row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  int row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  int row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  int row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  int row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  int row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  int row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];
#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {int col1Xreal = input[s][2*tx                                  ];	\
  int col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  int col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  int col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  int col2Xreal = input[s][2*tx + 1                              ];	\
  int col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  int col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  int col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  int row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  int row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  int row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  int row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  int row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  int row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  int row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  int row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];
#else
#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE

inline __device__ void dp4a(int &c, const int &a, const int &b) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
  char4 &a4 = *((char4*)&a);
  char4 &b4 = *((char4*)&b);
  c += a4.x*b4.x;
  c += a4.y*b4.y;
  c += a4.z*b4.z;
  c += a4.w*b4.w;
#endif
}

#define TWO_BY_TWO_COMPUTE(s)						\
  TWO_BY_TWO_PRELOAD(s)							\
  dp4a(sum11XXreal,row1Xreal,col1Xreal);				\
  dp4a(sum11XXreal,row1Ximag,col1Ximag);				\
  dp4a(sum11XXimag1,row1Ximag,col1Xreal);				\
  dp4a(sum11XXimag2,row1Xreal,col1Ximag);				\
  dp4a(sum11XYreal,row1Xreal,col1Yreal);				\
  dp4a(sum11XYreal,row1Ximag,col1Yimag);				\
  dp4a(sum11XYimag1,row1Ximag,col1Yreal);				\
  dp4a(sum11XYimag2,row1Xreal,col1Yimag);				\
  dp4a(sum11YXreal,row1Yreal,col1Xreal);				\
  dp4a(sum11YXreal,row1Yimag,col1Ximag);				\
  dp4a(sum11YXimag1,row1Yimag,col1Xreal);				\
  dp4a(sum11YXimag2,row1Yreal,col1Ximag);				\
  dp4a(sum11YYreal,row1Yreal,col1Yreal);				\
  dp4a(sum11YYreal,row1Yimag,col1Yimag);				\
  dp4a(sum11YYimag1,row1Yimag,col1Yreal);				\
  dp4a(sum11YYimag2,row1Yreal,col1Yimag);				\
  dp4a(sum12XXreal,row1Xreal,col2Xreal);				\
  dp4a(sum12XXreal,row1Ximag,col2Ximag);				\
  dp4a(sum12XXimag1,row1Ximag,col2Xreal);				\
  dp4a(sum12XXimag2,row1Xreal,col2Ximag);				\
  dp4a(sum12XYreal,row1Xreal,col2Yreal);				\
  dp4a(sum12XYreal,row1Ximag,col2Yimag);				\
  dp4a(sum12XYimag1,row1Ximag,col2Yreal);				\
  dp4a(sum12XYimag2,row1Xreal,col2Yimag);				\
  dp4a(sum12YXreal,row1Yreal,col2Xreal);				\
  dp4a(sum12YXreal,row1Yimag,col2Ximag);				\
  dp4a(sum12YXimag1,row1Yimag,col2Xreal);				\
  dp4a(sum12YXimag2,row1Yreal,col2Ximag);				\
  dp4a(sum12YYreal,row1Yreal,col2Yreal);				\
  dp4a(sum12YYreal,row1Yimag,col2Yimag);				\
  dp4a(sum12YYimag1,row1Yimag,col2Yreal);				\
  dp4a(sum12YYimag2,row1Yreal,col2Yimag);				\
  dp4a(sum21XXreal,row2Xreal,col1Xreal);				\
  dp4a(sum21XXreal,row2Ximag,col1Ximag);				\
  dp4a(sum21XXimag1,row2Ximag,col1Xreal);				\
  dp4a(sum21XXimag2,row2Xreal,col1Ximag);				\
  dp4a(sum21XYreal,row2Xreal,col1Yreal);				\
  dp4a(sum21XYreal,row2Ximag,col1Yimag);				\
  dp4a(sum21XYimag1,row2Ximag,col1Yreal);				\
  dp4a(sum21XYimag2,row2Xreal,col1Yimag);				\
  dp4a(sum21YXreal,row2Yreal,col1Xreal);				\
  dp4a(sum21YXreal,row2Yimag,col1Ximag);				\
  dp4a(sum21YXimag1,row2Yimag,col1Xreal);				\
  dp4a(sum21YXimag2,row2Yreal,col1Ximag);				\
  dp4a(sum21YYreal,row2Yreal,col1Yreal);				\
  dp4a(sum21YYreal,row2Yimag,col1Yimag);				\
  dp4a(sum21YYimag1,row2Yimag,col1Yreal);				\
  dp4a(sum21YYimag2,row2Yreal,col1Yimag);				\
  dp4a(sum22XXreal,row2Xreal,col2Xreal);				\
  dp4a(sum22XXreal,row2Ximag,col2Ximag);				\
  dp4a(sum22XXimag1,row2Ximag,col2Xreal);				\
  dp4a(sum22XXimag2,row2Xreal,col2Ximag);				\
  dp4a(sum22XYreal,row2Xreal,col2Yreal);				\
  dp4a(sum22XYreal,row2Ximag,col2Yimag);				\
  dp4a(sum22XYimag1,row2Ximag,col2Yreal);				\
  dp4a(sum22XYimag2,row2Xreal,col2Yimag);				\
  dp4a(sum22YXreal,row2Yreal,col2Xreal);				\
  dp4a(sum22YXreal,row2Yimag,col2Ximag);				\
  dp4a(sum22YXimag1,row2Yimag,col2Xreal);				\
  dp4a(sum22YXimag2,row2Yreal,col2Ximag);				\
  dp4a(sum22YYreal,row2Yreal,col2Yreal);				\
  dp4a(sum22YYreal,row2Yimag,col2Yimag);				\
  dp4a(sum22YYimag1,row2Yimag,col2Yreal);				\
  dp4a(sum22YYimag2,row2Yreal,col2Yimag);}
