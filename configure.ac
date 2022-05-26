# -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT(g6k-kernel, 0.1.2)
AC_CANONICAL_HOST
AC_CONFIG_SRCDIR([kernel/sieving.cpp])
AM_MAINTAINER_MODE([enable])

AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

#
# First extract pieces from the version number string
# 
G6K_MAJOR_VERSION=`echo AC_PACKAGE_VERSION | awk -F. '{print $1}'`
G6K_MINOR_VERSION=`echo AC_PACKAGE_VERSION | awk -F. '{print $2}'`
G6K_MICRO_VERSION=`echo AC_PACKAGE_VERSION | awk -F. '{print $3}'`
G6K_VERSION=$G6K_MAJOR_VERSION.$G6K_MINOR_VERSION.$G6K_MICRO_VERSION$G6K_MICRO_VERSION_SUFFIX
G6K_VERSION_NUMBER=`expr $G6K_MAJOR_VERSION \* 1000000 + $G6K_MINOR_VERSION \* 1000 + $G6K_MICRO_VERSION`

#
# Store version numbers in header
#
AC_DEFINE_UNQUOTED([G6K_MAJOR_VERSION],[$G6K_MAJOR_VERSION],[major version])
AC_DEFINE_UNQUOTED([G6K_MINOR_VERSION],[$G6K_MINOR_VERSION],[minor version])
AC_DEFINE_UNQUOTED([G6K_MICRO_VERSION],[$G6K_MICRO_VERSION],[micro version])
AC_DEFINE_UNQUOTED([G6K_VERSION],[$G6K_VERSION],[full version])
AC_DEFINE_UNQUOTED([G6K_VERSION_INFO],[$G6K_VERSION_INFO],[long version string])

#
# Libtool
# 
AC_PROG_LIBTOOL

# The version of the libtool library is of the form current:revision:age
#
# See http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
#
# When doing a release, they should be updated like this:
# 1. If no interfaces changed, only implementations: just increment
#    revision.
# 2. If interfaces were added, none removed: increment current, set
#    revision to zero and increment age.
# 3. If interfaces were removed (breaks backward compatibility): increment
#    current, and set both revision and age to zero.

G6K_LT_CURRENT=1
G6K_LT_REVISION=1
G6K_LT_AGE=0

AC_SUBST(G6K_LT_CURRENT)
AC_SUBST(G6K_LT_REVISION)
AC_SUBST(G6K_LT_AGE)


#
# Default flags.
#

DEFAULT_CXXFLAGS="-Ofast -ftree-vectorize -funroll-loops -Wall -Wextra"
CXXFLAGS=${CXXFLAGS:-$DEFAULT_CXXFLAGS}

#
# Checks for programs.
#
AC_LANG_CPLUSPLUS
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([11],[noext],[mandatory])

AX_PTHREAD

#
# Distributions may want to disable extensions in their build
#

AC_ARG_ENABLE(native,
AC_HELP_STRING([--enable-native],
               [build a binary optimized for the CPU found at compile time on systems that support it [default=yes]]),
               [case $enableval in
                  yes|no) ;;
                  *) AC_MSG_ERROR([bad value $enableval for --enable-native, need yes or no]) ;;
                esac],
               [enable_native=yes]
)

if test $enable_native = "yes"; then
  AX_CHECK_COMPILE_FLAG([-march=native], [], [enable_native=no])
  AX_EXT
  CXXFLAGS="${CXXFLAGS} ${SIMD_FLAGS}"
fi

#
# Template Dimensions
#

AC_ARG_ENABLE(templated-dim,
AC_HELP_STRING([--enable-templated-dim],
               [enable templated dimensions (longer compilation time, better performance) [default=no]]),
               [case $enableval in
                  yes|no) ;;
                  *) AC_MSG_ERROR([bad value $enableval for --enable-templated-dim, need yes or no]) ;;
                esac],
               [enable_templated_dim=no]
)

if test $enable_templated_dim = "yes"; then
     AC_DEFINE_UNQUOTED([TEMPLATED_DIM], 1, [enable templated dimensions])
fi

#
# Enable CPU counters
#

AC_ARG_ENABLE(cpucounters,
AC_HELP_STRING([--enable-cpucounters],
               [enable cpu performance counters [default=no]]),
               [case $enableval in
                  yes|no) ;;
                  *) AC_MSG_ERROR([bad value $enableval for --enable-cpucounters, need yes or no]) ;;
                esac],
               [enable_cpucounters=no]
)

if test $enable_cpucounters = "yes"; then
   CXXFLAGS="${CXXFLAGS} -DPERFORMANCE_COUNTING"
fi

#
# Enable stats
# 

enable_stats=0

AC_ARG_WITH(stats,
    AC_HELP_STRING([--with-stats@<:@=STATS@:>@], [enable statistics [supported=0,1,2; default=0]]),
    [enable_stats=$withval],
    [enable_stats=0]
)

if test $enable_stats = "1"; then
    AC_DEFINE_UNQUOTED([ENABLE_STATS], 1, [enable statistics])
elif test $enable_stats = "2"; then
    AC_DEFINE_UNQUOTED([ENABLE_EXTENDED_STATS], 1, [enable extended statistics])
fi

#
# Sieving parameters
#

AC_ARG_WITH(max-sieving-dim,
        AS_HELP_STRING([--with-max-sieving-dim@<:@=DIM@:>@], [maximum supported sieving dimension [default=128]]),
        [max_sieving_dim=$withval],
        [max_sieving_dim=128])

AC_DEFINE_UNQUOTED([MAX_SIEVING_DIM], $max_sieving_dim, [maximum supported sieving dimension])

AC_ARG_WITH(xpc-bucket-threshold,
        AS_HELP_STRING([--with-xpc-bucket-threshold@<:@=DIM@:>@], [XPC bucket threshold [default=102]]),
        [xpc_bucket_threshold=$withval],
        [xpc_bucket_threshold=102])

AC_DEFINE_UNQUOTED([XPC_BUCKET_THRESHOLD], $xpc_bucket_threshold, [popcount hamming weight threshold for buckets])

AC_ARG_WITH(xpc-threshold,
        AS_HELP_STRING([--with-xpc-threshold@<:@=DIM@:>@], [XPC threshold [default=96]]),
        [xpc_threshold=$withval],
        [xpc_threshold=96])

AC_DEFINE_UNQUOTED([XPC_THRESHOLD], $xpc_threshold, [popcount hamming weight threshold for pairs])

CXXFLAGS="${CXXFLAGS}"

AC_CONFIG_HEADER(kernel/g6k_config.h)
AC_CONFIG_FILES([Makefile
                 kernel/Makefile
                 g6k.pc])

AC_OUTPUT
