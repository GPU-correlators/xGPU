// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void FFT64_device( float2 *work )
{	
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
    
    //
    //  no sync in transpose is needed here if warpSize >= 32
    //  since the permutations are within-warp
    //
    
    float2 a[8];
    __shared__ float smem[64*9];
    
    load<8>( a, work, 64 );

    FFT8( a );
    
    twiddle<8>( a, lo, 64 );
    transpose_br<8>( a, &smem[hi*8*9+lo*9], 1, &smem[hi*8*9+lo], 9, 0 );
    
    FFT8( a );

    store<8>( a, work, 64 );
}	
    
extern "C" void FFT64( float2 *work, int batch )
{	
    FFT64_device<<< grid2D(batch/8), 64 >>>( work );
}	
