#ifndef OMP_XENGINE_H
#define OMP_XENGINE_H

#include "cuda_xengine.h" // For Complex and CompexInput typedefs

#ifdef __cplusplus
extern "C" {
#endif

void ompXengine(Complex *matrix_h, ComplexInput *array_h);

#ifdef __cplusplus
}
#endif

#endif // OMP_XENGINE_H
