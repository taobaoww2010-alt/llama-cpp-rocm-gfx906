if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
        if (EXISTS /opt/rocm)
            set(ROCM_PATH /opt/rocm)
        elseif (EXISTS /opt/rocm-6.3.0)
            set(ROCM_PATH /opt/rocm-6.3.0)
        else()
            set(ROCM_PATH /usr)
        endif()
    else()
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

find_package(rccl QUIET)

if (rccl_FOUND)
    message(STATUS "RCCL found - enabling NCCL API support")
    set(GGML_RCCL_NCCL_AVAILABLE ON)
else()
    message(STATUS "RCCL not found - using HIP P2P/Ring fallback")
    set(GGML_RCCL_NCCL_AVAILABLE OFF)
endif()

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

file(GLOB GGML_RCCL_HEADERS "*.h")
list(FILTER GGML_RCCL_HEADERS EXCLUDE REGEX ".*hip-comm\\.h$")

file(GLOB GGML_RCCL_SOURCES "*.cpp")

add_library(ggml-rccl OBJECT
            ${GGML_RCCL_HEADERS}
            ${GGML_RCCL_SOURCES})

target_include_directories(ggml-rccl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if (GGML_RCCL_NCCL_AVAILABLE)
    target_compile_definitions(ggml-rccl PUBLIC GGML_RCCL_NCCL_AVAILABLE=1)
else()
    target_compile_definitions(ggml-rccl PUBLIC GGML_RCCL_NCCL_AVAILABLE=0)
endif()

set_target_properties(ggml-rccl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET default
)

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_RCCL_SOURCES} PROPERTIES LANGUAGE CXX)
    if (rccl_FOUND)
        target_link_libraries(ggml-rccl PRIVATE rccl)
    endif()
    target_link_libraries(ggml-rccl PRIVATE hip::host)
else()
    set_source_files_properties(${GGML_RCCL_SOURCES} PROPERTIES LANGUAGE HIP)
endif()
