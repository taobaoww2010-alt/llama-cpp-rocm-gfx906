set(TENSOR_PARALLEL_SOURCES
    ggml-tensor-parallel.cpp
    ggml-tp-comm.cpp
)

add_library(ggml-tensor-parallel OBJECT ${TENSOR_PARALLEL_SOURCES})

target_include_directories(ggml-tensor-parallel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(GGML_HIP)
    target_compile_options(ggml-tensor-parallel PRIVATE
        -DGGML_USE_HIP
        $<$<CONFIG:Release>:-O3>
    )

    if(GGML_RCCL)
        target_compile_definitions(ggml-tensor-parallel PRIVATE GGML_USE_RCCL=1)
        target_link_libraries(ggml-tensor-parallel PRIVATE rccl)
    else()
        target_compile_definitions(ggml-tensor-parallel PRIVATE GGML_USE_RCCL=0)
    endif()

    hip_library_properties(ggml-tensor-parallel)
endif()
