// Define TEXTURE_DIM as 1 to use 1D texture (more accurate, costs 1 mult per LOAD)
// Define TEXTURE_DIM as 2 to use 2D texture (less accurate, saves 1 mult per LOAD)
#ifndef TEXTURE_DIM
#define TEXTURE_DIM 1
#endif

#if TEXTURE_DIM == 1
// Read in column in first warp as float2, row in second warp (still true for 1D?)
#define LOAD(s, t)							\
  {float2 temp = tex1Dfetch(tex1dfloat2, array_index + (t)*NFREQUENCY*Nstation*NPOL); \
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp; }

#elif TEXTURE_DIM == 2

//#define TEXTURE_FLOAT_COORD
#ifndef TEXTURE_FLOAT_COORD // use integer texture coordinates (requires ptx)

// Read float2 from global, write individual floats
// to shared memory avoid bank conflict.
#define LOAD(s, t)							\
  { float4 temp;							\
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [tex2dfloat2, {%4, %5}];" : \
	"=f"(temp.x), "=f"(temp.y), "=f"(temp.z), "=f"(temp.w) : "r"(array_index), "r"(t)); \
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = make_float2(temp.x, temp.y); }

#else // use float texture coordinates

// Read in column in first warp as float2, row in second warp
#define LOAD(s, t)							\
  {float2 temp = tex2D(tex2dfloat2, array_index, t);			\
    CUBE_ADD_BYTES(sizeof(ComplexInput));				\
    *(input##s##_p) = temp; }

#endif // TEXTURE_FLOAT_COORD

#else
#error TEXTURE_DIM must be 1 or 2
#endif

// read in shared data as individual floats to avoid bank conflicts

#define TWO_BY_TWO_COMPUTE(s) {						\
    float2 col1X = input[s][4*tx + 0];					\
    float2 col1Y = input[s][4*tx + 1];					\
    float2 row1X = input[s][4*ty + 0 + 4*TILE_WIDTH];			\
    float2 row1Y = input[s][4*ty + 1 + 4*TILE_WIDTH];			\
    float2 col2X = input[s][4*tx + 2];					\
    float2 col2Y = input[s][4*tx + 3];					\
    float2 row2X = input[s][4*ty + 2 + 4*TILE_WIDTH];			\
    float2 row2Y = input[s][4*ty + 3 + 4*TILE_WIDTH];			\
    sum11XXreal += row1X.x * col1X.x;					\
    sum11XXimag += row1X.y * col1X.x;					\
    sum11XXreal += row1X.y * col1X.y;					\
    sum11XXimag -= row1X.x * col1X.y;					\
    sum11XYreal += row1X.x * col1Y.x;					\
    sum11XYimag += row1X.y * col1Y.x;					\
    sum11XYreal += row1X.y * col1Y.y;					\
    sum11XYimag -= row1X.x * col1Y.y;					\
    sum11YYreal += row1Y.x * col1Y.x;					\
    sum11YYimag += row1Y.y * col1Y.x;					\
    sum11YYreal += row1Y.y * col1Y.y;					\
    sum11YYimag -= row1Y.x * col1Y.y;					\
    sum11YXreal += row1Y.y * col1X.y;					\
    sum11YXimag += row1Y.y * col1X.x;					\
    sum11YXreal += row1Y.x * col1X.x;					\
    sum11YXimag -= row1Y.x * col1X.y;					\
    sum12XXreal += row1X.x * col2X.x;					\
    sum12XXimag += row1X.y * col2X.x;					\
    sum12XXreal += row1X.y * col2X.y;					\
    sum12XXimag -= row1X.x * col2X.y;					\
    sum12XYreal += row1X.x * col2Y.x;					\
    sum12XYimag += row1X.y * col2Y.x;					\
    sum12XYreal += row1X.y * col2Y.y;					\
    sum12XYimag -= row1X.x * col2Y.y;					\
    sum12YYreal += row1Y.x * col2Y.x;					\
    sum12YYimag += row1Y.y * col2Y.x;					\
    sum12YYreal += row1Y.y * col2Y.y;					\
    sum12YYimag -= row1Y.x * col2Y.y;					\
    sum12YXreal += row1Y.x * col2X.x;					\
    sum12YXimag += row1Y.y * col2X.x;					\
    sum12YXreal += row1Y.y * col2X.y;					\
    sum12YXimag -= row1Y.x * col2X.y;					\
    sum22XXreal += row2X.x * col2X.x;					\
    sum22XXimag += row2X.y * col2X.x;					\
    sum22XXreal += row2X.y * col2X.y;					\
    sum22XXimag -= row2X.x * col2X.y;					\
    sum22XYreal += row2X.x * col2Y.x;					\
    sum22XYimag += row2X.y * col2Y.x;					\
    sum22XYreal += row2X.y * col2Y.y;					\
    sum22XYimag -= row2X.x * col2Y.y;					\
    sum22YYreal += row2Y.x * col2Y.x;					\
    sum22YYimag += row2Y.y * col2Y.x;					\
    sum22YYreal += row2Y.y * col2Y.y;					\
    sum22YYimag -= row2Y.x * col2Y.y;					\
    sum22YXreal += row2Y.x * col2X.x;					\
    sum22YXimag += row2Y.y * col2X.x;					\
    sum22YXreal += row2Y.y * col2X.y;					\
    sum22YXimag -= row2Y.x * col2X.y;					\
    sum21XXreal += row2X.x * col1X.x;					\
    sum21XXimag += row2X.y * col1X.x;					\
    sum21XXreal += row2X.y * col1X.y;					\
    sum21XXimag -= row2X.x * col1X.y;					\
    sum21XYreal += row2X.x * col1Y.x;					\
    sum21XYimag += row2X.y * col1Y.x;					\
    sum21XYreal += row2X.y * col1Y.y;					\
    sum21XYimag -= row2X.x * col1Y.y;					\
    sum21YYreal += row2Y.x * col1Y.x;					\
    sum21YYimag += row2Y.y * col1Y.x;					\
    sum21YYreal += row2Y.y * col1Y.y;					\
    sum21YYimag -= row2Y.x * col1Y.y;					\
    sum21YXreal += row2Y.x * col1X.x;					\
    sum21YXimag += row2Y.y * col1X.x;					\
    sum21YXreal += row2Y.y * col1X.y;					\
    sum21YXimag -= row2Y.x * col1X.y;					\
  }

