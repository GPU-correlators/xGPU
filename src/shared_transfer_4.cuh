// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  {float2 temp = tex1Dfetch(tex1dfloat2, array_index + (t)*NFREQUENCY*Nstation*NPOL);			\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}


#elif TEXTURE_DIM == 2

//#define TEXTURE_FLOAT_COORD
#ifndef TEXTURE_FLOAT_COORD // use integer texture coordinates (requires ptx)

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  {  float4 temp;							\
  asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [tex2dfloat2, {%4, %5}];" :	\
      "=f"(temp.x), "=f"(temp.y), "=f"(temp.z), "=f"(temp.w) : "r"(array_index), "r"(t)); \
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}

#else  // use float texture coordinates

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  { float2 temp = tex2D(tex2dfloat2, array_index, t);			\
    *(input##s##_p) = temp.x;						\
    *(input##s##_p + 4*TILE_WIDTH) = temp.y;}

#endif // TEXTURE_FLOAT_COORD

#else
#error TEXTURE_DIM must be 1 or 2
#endif

// read in shared data as individual floats to avoid bank conflicts

#if COMPLEX_BLOCK_SIZE == 1
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][4*tx                                   ];	\
  float col1Ximag = input[s][4*tx     + 4*TILE_WIDTH                ];	\
  float col1Yreal = input[s][4*tx + 1                               ];	\
  float col1Yimag = input[s][4*tx + 1 + 4*TILE_WIDTH                ];	\
  float col2Xreal = input[s][4*tx + 2                               ];	\
  float col2Ximag = input[s][4*tx + 2 + 4*TILE_WIDTH                ];	\
  float col2Yreal = input[s][4*tx + 3                               ];	\
  float col2Yimag = input[s][4*tx + 3 + 4*TILE_WIDTH                ];	\
  float row1Xreal = input[s][4*ty                     + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][4*ty     + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][4*ty + 1                 + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][4*ty + 1 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][4*ty + 2                 + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][4*ty + 2 + 4*TILE_HEIGHT + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][4*ty + 3                 + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][4*ty + 3 + 4*TILE_HEIGHT + 8*TILE_WIDTH];
#elif COMPLEX_BLOCK_SIZE == 32
#define TWO_BY_TWO_PRELOAD(s)						\
 {float col1Xreal = input[s][2*tx                                  ];	\
  float col1Ximag = input[s][2*tx     + 2*TILE_WIDTH               ];	\
  float col1Yreal = input[s][2*tx     + 4*TILE_WIDTH               ];	\
  float col1Yimag = input[s][2*tx     + 6*TILE_WIDTH               ];	\
  float col2Xreal = input[s][2*tx + 1                              ];	\
  float col2Yreal = input[s][2*tx + 1 + 4*TILE_WIDTH               ];	\
  float col2Ximag = input[s][2*tx + 1 + 2*TILE_WIDTH               ];	\
  float col2Yimag = input[s][2*tx + 1 + 6*TILE_WIDTH               ];	\
  float row1Xreal = input[s][2*ty                    + 8*TILE_WIDTH];	\
  float row1Ximag = input[s][2*ty     + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yreal = input[s][2*ty     + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row1Yimag = input[s][2*ty     + 6*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Xreal = input[s][2*ty + 1                + 8*TILE_WIDTH];	\
  float row2Yreal = input[s][2*ty + 1 + 4*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Ximag = input[s][2*ty + 1 + 2*TILE_WIDTH + 8*TILE_WIDTH];	\
  float row2Yimag = input[s][2*ty + 1 + 6*TILE_WIDTH + 8*TILE_WIDTH];
#else
#error COMPLEX_BLOCK_SIZE must be 1 or 32
#endif // COMPLEX_BLOCK_SIZE

#define TWO_BY_TWO_COMPUTE(s)						\
  TWO_BY_TWO_PRELOAD(s)							\
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
