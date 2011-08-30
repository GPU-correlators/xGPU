// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California. 
// All rights reserved.

#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cufft.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>

#define TIMER_TOLERANCE 0.1f

#define BEGIN_TIMING( )	\
{\
    unsigned int n_iterations;	\
    for( n_iterations = 1; n_iterations < 0x80000000; n_iterations *= 2 )\
    {\
        Q( cudaThreadSynchronize( ) );\
        Q( cudaEventRecord( start, 0 ) );\
        for( unsigned int iteration = 0; iteration < n_iterations; iteration++ ){

#define END_TIMING( seconds ) }\
        Q( cudaEventRecord( end, 0 ) );\
        Q( cudaEventSynchronize( end ) );\
        float milliseconds;\
        Q( cudaEventElapsedTime( &milliseconds, start, end ) );\
        seconds = milliseconds/1e3f;\
        if( seconds >= TIMER_TOLERANCE )\
            break;\
    }\
    seconds /= n_iterations;\
}

#define Q( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

extern "C" void FFT8( float2 *work, int batch );
extern "C" void FFT16( float2 *work, int batch );
extern "C" void FFT64( float2 *work, int batch );
extern "C" void FFT256( float2 *work, int batch );
extern "C" void FFT512( float2 *work, int batch );
extern "C" void FFT1024( float2 *work, int batch );
extern "C" void FFT2048( float2 *work, int batch );
extern "C" void FFT4096( float2 *work, int batch );
extern "C" void FFT8192( float2 *work, int batch );
extern "C" void IFFT8( float2 *work, int batch );
extern "C" void IFFT16( float2 *work, int batch );
extern "C" void IFFT64( float2 *work, int batch );
extern "C" void IFFT256( float2 *work, int batch );
extern "C" void IFFT512( float2 *work, int batch );
extern "C" void IFFT1024( float2 *work, int batch );
extern "C" void IFFT2048( float2 *work, int batch );
extern "C" void IFFT4096( float2 *work, int batch );
extern "C" void IFFT8192( float2 *work, int batch );
typedef void (*FFT_t)( float2 *work, int batch );

const float ulp =  1.192092896e-07f;

inline float max( float a, float b ) { return a > b ? a : b; }

#ifndef _MSC_VER
#define _isnan(a) (fpclassify(a) == FP_NAN)
#endif

inline double2 operator*( double2 a, double2 b ) { return make_double2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline double2 operator+( double2 a, double2 b ) { return make_double2( a.x + b.x, a.y + b.y ); }
inline double2 operator-( double2 a, double2 b ) { return make_double2( a.x - b.x, a.y - b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float norm2( float2 a ) { return a.x*a.x+a.y*a.y ; }

//  
//	Implementation of Stockham's FFT as in:
//  
//	Bailey, D. H. 1988. A High-Performance FFT Algorithm for Vector 
//     Supercomputers, International Journal of Supercomputer 
//     Applications 2, 1, 82--87. (Available online at
//     http://crd.lbl.gov/~dhbailey/dhbpapers/fftzp.pdf)
//  
void compute_reference( float2 *dst, float2 *src, int n, int batch )
{   
    double2 *X = (double2*) malloc( n*sizeof(double2) );
    double2 *Y = (double2*) malloc( n*sizeof(double2) );
    for( int ibatch = 0; ibatch < batch; ibatch++ )
    {
        // go to double precision
        for( int i = 0; i < n; i++ )
            X[i] = make_double2( src[i].x, src[i].y );
        
        // FFT in double precision
        for( int kmax = 1, jmax = n/2; kmax < n; kmax *= 2, jmax /= 2 )
        {
            for( int k = 0; k < kmax; k++ )
            {
                double phi = -2.*M_PI*k/(2.*kmax);
                double2 w = make_double2( cos(phi), sin(phi) ); 
                for( int j = 0; j < jmax; j++ )
                {
                    Y[j*2*kmax + k]        = X[j*kmax + k] + w * X[j*kmax + n/2 + k];
                    Y[j*2*kmax + kmax + k] = X[j*kmax + k] - w * X[j*kmax + n/2 + k];
                }
            }
            double2 *T = X;
            X = Y;
            Y = T;
        }
        
        // return to single precision
        for( int i = 0; i < n; i++ )
            dst[i] = make_float2( (float)X[i].x, (float)X[i].y );
        
        src += n;
        dst += n;
    }
    free( X );
    free( Y );
}   
    
//  
//	The relative forward error is bound by ~logN, see Th. 24.2 in:
//  
//  Higham, N. J. 2002. Accuracy and Stability of Numerical Algorithms, SIAM.
//    (Available online at http://books.google.com/books?id=epilvM5MMxwC)
//  
float relative_error( float2 *reference, float2 *result, int n, int batch )
{   
    float error = 0;
    for( int i = 0; i < batch; i++ )
    {
        float diff = 0, norm = 0;
        for( int j = 0; j < n; j++ )
        {
            diff += norm2( reference[j] - result[j] );
            norm += norm2( reference[j] );
        }
        if( _isnan( diff ) )
            return -1;
        
        error = max( error, diff / norm );
        
        reference += n;
        result += n;
    }
    return sqrt( error ) / ulp;
}   
    
void transpose( float2 *dst, float2 *src, int num, int sz, int sy, int sx )
{	
    const int recordsize = sx*sy*sz;

    float2 *temp = (float2 *)malloc( recordsize * sizeof(float2) );
    for( int irecord = 0; irecord < num; irecord += recordsize )
    {
        //transpose one record to a buffer
        for( int z = 0; z < sz; z++ )
            for( int y = 0; y < sy; y++ )
                memcpy( temp + sx*(z+y*sz), src + sx*(z*sy+y), sizeof(float2)*sx );
        
        //copy result back
        memcpy( dst, temp, sizeof(float2)*recordsize );

        //repeat with next record
        dst += recordsize;
        src += recordsize;
    }
    free( temp );
}   
    
//  
//  MAIN
//  
int main( int argc, char **argv )
{	
    int n_entries = 8*1024*1024;
    int n_bytes = n_entries * sizeof(float2);
    
    int idevice = 0;
    for( int i = 1; i < argc-1; i ++ )
        if( strcmp( argv[i], "-device" ) == 0 )
            idevice = atoi( argv[i+1] );
    
    Q( cudaSetDevice( idevice ) );
    
    struct cudaDeviceProp prop;
    Q( cudaGetDeviceProperties( &prop, idevice ) );
    printf( "\nDevice: %s, %.0f MHz clock, %.0f MB memory.\n", prop.name, prop.clockRate/1000.f, prop.totalGlobalMem/1024.f/1024.f );
    printf( "Compiled with CUDA %d.\n", CUDART_VERSION );
    
    cufftHandle plan;
    cudaEvent_t start, end;
    Q( cudaEventCreate( &start ) );
    Q( cudaEventCreate( &end ) );
    
    float2 *work;
    Q( cudaMalloc( (void**)&work, n_bytes ) );
    
    float2 *source    = (float2 *)malloc( n_bytes );
    float2 *result    = (float2 *)malloc( n_bytes );
    float2 *reference = (float2 *)malloc( n_bytes );
    
    for( int i = 0; i < n_entries; i++ )
    {
        source[i].x = (rand()/(float)RAND_MAX)*2-1;
        source[i].y = (rand()/(float)RAND_MAX)*2-1;
    }
    
    //
    //	main loop
    //
    int ns[] = { 8, 16, 64, 256, 512, 1024, 2048, 4096, 8192 };
    FFT_t FFTs[] = { FFT8, FFT16, FFT64, FFT256, FFT512, FFT1024, FFT2048, FFT4096, FFT8192 };
    FFT_t IFFTs[] = { IFFT8, IFFT16, IFFT64, IFFT256, IFFT512, IFFT1024, IFFT2048, IFFT4096, IFFT8192 };
    printf( "             --------CUFFT-------  ---This prototype---  ---two way---\n" );
    printf( "   N   Batch Gflop/s  GB/s  error  Gflop/s  GB/s  error  Gflop/s error\n" );
    for( int in = 0; in < sizeof(ns)/sizeof(ns[0]); in++ )
    {
        int n = ns[in];
        int batch = n_entries / n;
        FFT_t FFT = FFTs[in];
        FFT_t IFFT = IFFTs[in];
        
        float s, ulps;
        float Gflop = 5e-9f * n * logf((float)n) / logf(2) * batch;
        float GB = 2e-9f * n * batch * sizeof(float2);
        
        printf( "%4d %7d", n, batch );
        
        compute_reference( reference, source, n, batch );
        
        //  
        //  run CUFFT
        //  
        Q( cufftPlan1d( &plan, n, CUFFT_C2C, batch ) );
        
        //  upload / compute / download
        Q( cudaMemcpy( work, source, n_bytes, cudaMemcpyHostToDevice ) );
        Q( cufftExecC2C( plan, work, work, CUFFT_FORWARD ) );
        Q( cudaMemcpy( result, work, n_bytes, cudaMemcpyDeviceToHost ) );
        
        ulps = relative_error( reference, result, n, batch );
        
        //  time
        Q( cufftExecC2C( plan, work, work, CUFFT_FORWARD ) );
        BEGIN_TIMING( );
            Q( cufftExecC2C( plan, work, work, CUFFT_FORWARD ) );
        END_TIMING( s );
        
        Q( cufftDestroy( plan ) );
        
        printf( " %6.1f %6.1f %5.1f ", Gflop/s, GB/s, ulps );

        //  
        //  run the prototype
        //  
        if( n >= 256 )
        {
            //  upload / compute / download
            Q( cudaMemcpy( work, source, n_bytes, cudaMemcpyHostToDevice ) );
            FFT( work, batch );
            Q( cudaMemcpy( result, work, n_bytes, cudaMemcpyDeviceToHost ) );
            if( n >= 2048 )
                transpose( result, result, n*batch, n/512, 512, 1 );
        }
        else
        {
            //  transpose / upload / compute / download / transpose
            if( n == 16 )
                transpose( result, source, n*batch, 64, 16, 1 );
            else
                transpose( result, source, n*batch, 512/n, 8, n/8 );
            
            Q( cudaMemcpy( work, result, n_bytes, cudaMemcpyHostToDevice ) );
            FFT( work, batch );
            Q( cudaMemcpy( result, work, n_bytes, cudaMemcpyDeviceToHost ) );
            
            if( n == 16 )
                transpose( result, result, n*batch, 16, 64, 1 );
            else
                transpose( result, result, n*batch, 8, 512/n, n/8 );
        }
        
        ulps = relative_error( reference, result, n, batch );
        
        //  time
        FFT( work, batch );
        BEGIN_TIMING( );
            FFT( work, batch );
        END_TIMING( s );

        printf( " %7.1f %6.1f %5.1f", Gflop/s, GB/s, ulps );
        
        //test forward+inverse
        Q( cudaMemcpy( work, source, n_bytes, cudaMemcpyHostToDevice ) );
        FFT( work, batch );
        IFFT( work, batch );
        Q( cudaMemcpy( result, work, n_bytes, cudaMemcpyDeviceToHost ) );
        //normalize output
        for( int i = 0; i < n_entries; i++ )
        {
            result[i].x /= n;
            result[i].y /= n;
        }
        //find error
        double ulps2 = relative_error( source, result, n, batch );
        
        //  time
        double s2;
        FFT( work, batch );
        IFFT( work, batch );
        BEGIN_TIMING( );
            FFT( work, batch );
            IFFT( work, batch );
        END_TIMING( s2 );

        printf( "  %7.1f %5.1f\n", 2*Gflop/s2, ulps2 );
    }
    
    printf( "\nErrors are supposed to be of order of 1 (ULPs).\n\n" );
    
    //
    //	release resources
    //
    Q( cudaEventDestroy( start ) );
    Q( cudaEventDestroy( end ) );
    Q( cudaFree( work ) );
    free( source );
    free( result );
    free( reference );
}
