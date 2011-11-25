#ifndef CPU_UTIL_H
#define CPU_UTIL_H

// Functions in cpu_util.cc

void random_complex(ComplexInput* random_num, int length);

void reorderMatrix(Complex *matrix);

void checkResult(Complex *gpu, Complex *cpu, int verbose=0, ComplexInput *array_h=0);

void extractMatrix(Complex *matrix, Complex *packed);

#endif // CPU_UTIL_H
