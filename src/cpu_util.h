#ifndef CPU_UTIL_H
#define CPU_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

// Functions in cpu_util.cc

void random_complex(ComplexInput* random_num, int length);

void reorderMatrix(Complex *matrix);

void checkResult(Complex *gpu, Complex *cpu, int verbose, ComplexInput *array_h);

void extractMatrix(Complex *matrix, Complex *packed);

#ifdef __cplusplus
}
#endif

#endif // CPU_UTIL_H
