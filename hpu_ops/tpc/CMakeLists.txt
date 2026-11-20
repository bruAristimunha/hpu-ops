cmake_minimum_required(VERSION 3.16)
project(linalg_tpc_kernels CXX C)

set(TPC_COMPILER_PATH "/usr/bin/tpc-clang" CACHE STRING "TPC compiler")
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "TPC includes")

add_library(linalg_tpc SHARED glue.cpp)
target_include_directories(linalg_tpc PRIVATE ${TPC_INCLUDE_DIR})

# gj_solve (A, B) and gj_inv (A alone, B = identity, -DIDENTITY_B) share one source;
# jacobi_eigh and jacobi_eigh_refine (Pass 2 adaptive-shift refinement, -DREFINE_PASS) share one source.
foreach(KERNEL gj_solve_f32_gaudi2 gj_inv_f32_gaudi2 jacobi_eigh_f32_gaudi2 jacobi_eigh_refine_f32_gaudi2)
    set(SOURCE ${KERNEL})
    set(DEFINES "")
    if(KERNEL STREQUAL "gj_inv_f32_gaudi2")
        set(SOURCE gj_solve_f32_gaudi2)
        set(DEFINES -DIDENTITY_B)
    elseif(KERNEL STREQUAL "jacobi_eigh_refine_f32_gaudi2")
        set(SOURCE jacobi_eigh_f32_gaudi2)
        set(DEFINES -DREFINE_PASS)
    endif()
    add_custom_command(
        OUTPUT ${KERNEL}_x86.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.c
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.c ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.c
        COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=gaudi2 -O2 ${DEFINES} -I${TPC_INCLUDE_DIR}/ ./${KERNEL}.c -c
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${KERNEL}.o ${KERNEL}_x86.o)
    add_library(${KERNEL} STATIC ${KERNEL}_x86.o)
    set_target_properties(${KERNEL} PROPERTIES LINKER_LANGUAGE C)
    target_link_libraries(linalg_tpc ${KERNEL})
endforeach()

# One source, three kernels: jacobi_svd (thin), jacobi_svd_full (-DFULL_U), and jacobi_svdvals (S only, -DNO_UV).
set(SVD_SOURCE jacobi_svd_f32_gaudi2)
foreach(KERNEL jacobi_svd_f32_gaudi2 jacobi_svd_full_f32_gaudi2 jacobi_svdvals_f32_gaudi2)
    set(DEFINES "")
    if(KERNEL STREQUAL "jacobi_svd_full_f32_gaudi2")
        set(DEFINES -DFULL_U)
    elseif(KERNEL STREQUAL "jacobi_svdvals_f32_gaudi2")
        set(DEFINES -DNO_UV)
    endif()
    add_custom_command(
        OUTPUT ${KERNEL}_x86.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SVD_SOURCE}.c
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${SVD_SOURCE}.c ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.c
        COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=gaudi2 -O2 ${DEFINES} -I${TPC_INCLUDE_DIR}/ ./${KERNEL}.c -c
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${KERNEL}.o ${KERNEL}_x86.o)
    add_library(${KERNEL} STATIC ${KERNEL}_x86.o)
    set_target_properties(${KERNEL} PROPERTIES LINKER_LANGUAGE C)
    target_link_libraries(linalg_tpc ${KERNEL})
endforeach()

# lu_factor: packed LU with LAPACK pivots, the permutation and info; lu_perm (-DPERM_FROM_PIVOTS, same source):
# pivots to permutation; lu_solve: getrs from the packed factors and pivots.
foreach(KERNEL householder_product_f32_gaudi2 lu_factor_f32_gaudi2 lu_perm_f32_gaudi2 lu_solve_f32_gaudi2)
    set(SOURCE ${KERNEL})
    set(DEFINES "")
    if(KERNEL STREQUAL "lu_perm_f32_gaudi2")
        set(SOURCE lu_factor_f32_gaudi2)
        set(DEFINES -DPERM_FROM_PIVOTS)
    endif()

    add_custom_command(
        OUTPUT ${KERNEL}_x86.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.c
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}.c ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.c
        COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=gaudi2 -O2 ${DEFINES} -I${TPC_INCLUDE_DIR}/ ./${KERNEL}.c -c
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${KERNEL}.o ${KERNEL}_x86.o)
    add_library(${KERNEL} STATIC ${KERNEL}_x86.o)
    set_target_properties(${KERNEL} PROPERTIES LINKER_LANGUAGE C)
    target_link_libraries(linalg_tpc ${KERNEL})
endforeach()

# One source, two kernels: matrix_exp (A) and matrix_exp_frechet (A, G -> L_exp(A^T, G), -DFRECHET).
# matrix_exp_frechet uses -O1: tpc-clang 14.0.5 -O2 miscompiles the __local__ array copy after matmul_nn.
set(EXPM_SOURCE matrix_exp_f32_gaudi2)
foreach(KERNEL matrix_exp_f32_gaudi2 matrix_exp_frechet_f32_gaudi2)
    set(DEFINES "")
    set(OPT "-O2")
    if(KERNEL STREQUAL "matrix_exp_frechet_f32_gaudi2")
        set(DEFINES -DFRECHET)
        set(OPT "-O1")
    endif()
    add_custom_command(
        OUTPUT ${KERNEL}_x86.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${EXPM_SOURCE}.c
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${EXPM_SOURCE}.c ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.c
        COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=gaudi2 ${OPT} ${DEFINES} -I${TPC_INCLUDE_DIR}/ ./${KERNEL}.c -c
        COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${KERNEL}.o ${KERNEL}_x86.o)
    add_library(${KERNEL} STATIC ${KERNEL}_x86.o)
    set_target_properties(${KERNEL} PROPERTIES LINKER_LANGUAGE C)
    target_link_libraries(linalg_tpc ${KERNEL})
endforeach()
