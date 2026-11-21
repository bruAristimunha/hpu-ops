cmake_minimum_required(VERSION 3.16)
project(linalg_tpc_kernels CXX C)

set(TPC_COMPILER_PATH "/usr/bin/tpc-clang" CACHE STRING "TPC compiler")
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "TPC includes")

add_library(linalg_tpc SHARED glue.cpp)
target_include_directories(linalg_tpc PRIVATE ${TPC_INCLUDE_DIR})

# gj_solve (A, B) and gj_inv (A alone, B = identity, -DIDENTITY_B) share one source;
# jacobi_eigh and jacobi_eigh_refine (Pass 2 adaptive-shift refinement, -DREFINE_PASS) share one source;
# the 4 tri_solve variants share tri_solve_f32_gaudi2.c (-DUPPER, -DUNIT_DIAG).
foreach(KERNEL
        gj_solve_f32_gaudi2 gj_inv_f32_gaudi2 jacobi_eigh_f32_gaudi2 jacobi_eigh_refine_f32_gaudi2
        tri_solve_lower_f32_gaudi2 tri_solve_upper_f32_gaudi2
        tri_solve_lower_unit_f32_gaudi2 tri_solve_upper_unit_f32_gaudi2)
    set(SOURCE ${KERNEL})
    set(DEFINES "")
    if(KERNEL STREQUAL "gj_inv_f32_gaudi2")
        set(SOURCE gj_solve_f32_gaudi2)
        set(DEFINES -DIDENTITY_B)
    elseif(KERNEL MATCHES "^tri_solve_")
        set(SOURCE tri_solve_f32_gaudi2)
        if(KERNEL MATCHES "_upper_")
            list(APPEND DEFINES -DUPPER)
        endif()
        if(KERNEL MATCHES "_unit_")
            list(APPEND DEFINES -DUNIT_DIAG)
        endif()
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
