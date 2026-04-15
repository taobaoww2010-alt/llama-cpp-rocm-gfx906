cmake_minimum_required(VERSION 3.21)
project(test_rccl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED ROCM_PATH)
    if (EXISTS /opt/rocm-6.3.0)
        set(ROCM_PATH /opt/rocm-6.3.0)
    elseif (EXISTS /opt/rocm)
        set(ROCM_PATH /opt/rocm)
    else()
        set(ROCM_PATH /usr)
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

find_package(HIP REQUIRED)
find_package(rccl REQUIRED)

include_directories(
    ${HIP_INCLUDE_DIRS}
    ${ROCM_PATH}/include
)

add_executable(test_rccl test_rccl.cpp)

target_link_libraries(test_rccl
    PRIVATE
    rccl
    hip::hip
)

target_compile_options(test_rccl PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-x hip --hip-link>
)
