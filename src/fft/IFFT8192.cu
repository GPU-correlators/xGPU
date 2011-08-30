// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void IFFT512_device( float2 *work );

#define rank 16
__global__ void IFFT16_device_( float2 *work )
{	
    int tid = threadIdx.x;
    
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int lo = bid & (8192/rank/64-1);
    int hi = bid &~(8192/rank/64-1);
    
    int i = lo*64 + tid;
    
    work += hi * (rank*64) + i;
    
    float2 a[rank];
    load<rank>( a, work, 512 );
    itwiddle_straight<rank>( a, i, 8192 );
    IFFT16( a );
    store<rank>( a, work, 512 );
}	

extern "C" void IFFT8192( float2 *work, int batch )
{	
    IFFT512_device<<< grid2D(batch*rank), 64 >>>( work );
    IFFT16_device_<<< grid2D(batch*(8192/rank)/64), 64 >>>( work );
}	
