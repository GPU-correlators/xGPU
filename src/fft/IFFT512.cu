// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void IFFT512_device( float2 *work )
{	
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;
    
    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
	
    float2 a[8];
    __shared__ float smem[8*8*9];
    
    load<8>( a, work, 64 );

    IFFT8( a );
	
    itwiddle<8>( a, tid, 512 );
    transpose<8>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );
	
    IFFT8( a );
	
    itwiddle<8>( a, hi, 64);
    transpose<8>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );
    
    IFFT8( a );

    store<8>( a, work, 64 );
}	
    
extern "C" void IFFT512( float2 *work, int batch )
{	
    IFFT512_device<<< grid2D(batch), 64 >>>( work );
}	
