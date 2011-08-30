// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include "codelets.h"

__global__ void IFFT512_device( float2 *work );

#define N 2048
#define rank 4
__global__ void IFFT4_device_( float2 *work )
{	
    int tid = threadIdx.x;
    
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int lo = bid & (N/rank/64-1);
    int hi = bid &~(N/rank/64-1);
    
    int i = lo*64 + tid;
    
    work += hi * (rank*64) + i;
    
    float2 a[rank];
    load<rank>( a, work, 512 );
    itwiddle_straight<rank>( a, i, N );
    IFFT4( a );
    store<rank>( a, work, 512 );
}	

extern "C" void IFFT2048( float2 *work, int batch )
{	
    IFFT512_device<<< grid2D(batch*rank), 64 >>>( work );
    IFFT4_device_<<< grid2D(batch*(N/rank)/64), 64 >>>( work );
}	
