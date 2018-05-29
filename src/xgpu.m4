# serial 1 xgpu.m4
AC_DEFUN([AX_CHECK_XGPU],
[AC_PREREQ([2.65])dnl
AC_ARG_WITH([xgpu],
            AC_HELP_STRING([--with-xgpu=DIR],
                           [Location of xGPU headers/libs (/usr/local)]),
            [XGPUDIR="$withval"],
            [XGPUDIR=/usr/local])

orig_LDFLAGS="${LDFLAGS}"
LDFLAGS="${orig_LDFLAGS} -L${XGPUDIR}/lib"
AC_CHECK_LIB([xgpu], [xgpuInit],
             # Found
             AC_SUBST(XGPU_LIBDIR,${XGPUDIR}/lib),
             # Not found there, check XGPUDIR
             AS_UNSET(ac_cv_lib_xgpu_xgpuInit)
             LDFLAGS="${orig_LDFLAGS} -L${XGPUDIR}"
             AC_CHECK_LIB([XGPU], [xgpuInit],
                          # Found
                          AC_SUBST(XGPU_LIBDIR,${XGPUDIR}),
                          # Not found there, error
                          AC_MSG_ERROR([xGPU library not found])))
LDFLAGS="${orig_LDFLAGS}"

AC_CHECK_FILE([${XGPUDIR}/include/xgpu.h],
              # Found
              AC_SUBST(XGPU_INCDIR,${XGPUDIR}/include),
              # Not found there, check XGPUDIR
              AC_CHECK_FILE([${XGPUDIR}/xgpu.h],
                            # Found
                            AC_SUBST(XGPU_INCDIR,${XGPUDIR}),
                            # Not found there, error
                            AC_MSG_ERROR([XGPU.h header file not found])))
])

dnl Calls AX_CHECK_XGPU and then checks for and uses xgpuinfo to define the
dnl following macros in config.h:
dnl
dnl   XGPU_NSTATION   - Number of dual-pol(!) stations per xGPU instance
dnl   XGPU_NFREQUENCY - Number of frequency channels per xGPU instance
dnl   XGPU_NTIME      - Number of time samples per freqency channel per xGPU
dnl                     instance
dnl
AC_DEFUN([AX_CHECK_XGPUINFO],
[AC_PREREQ([2.65])dnl
AX_CHECK_XGPU
AC_CHECK_FILE([${XGPUDIR}/bin/xgpuinfo],
              # Found
              AC_SUBST(XGPU_BINDIR,${XGPUDIR}/bin),
              # Not found there, check XGPUDIR
              AC_CHECK_FILE([${XGPUDIR}/xgpuinfo],
                            # Found
                            AC_SUBST(XGPU_BINDIR,${XGPUDIR}),
                            # Not found there, error
                            AC_MSG_ERROR([xgpuinfo program not found])))

AC_DEFINE_UNQUOTED([XGPU_NSTATION],
                   [`${XGPU_BINDIR}/xgpuinfo | sed -n '/Number of stations: /{s/.*: //;p}'`],
                   [Number of stations == Ninputs/2])

AC_DEFINE_UNQUOTED([XGPU_NFREQUENCY],
                   [`${XGPU_BINDIR}/xgpuinfo | sed -n '/Number of frequencies: /{s/.*: //;p}'`],
                   [Number of frequency channels per xGPU instance])

AC_DEFINE_UNQUOTED([XGPU_NTIME],
                   [`${XGPU_BINDIR}/xgpuinfo | sed -n '/time samples per GPU integration: /{s/.*: //;p}'`],
                   [Number of time samples (i.e. spectra) per xGPU integration])
])
