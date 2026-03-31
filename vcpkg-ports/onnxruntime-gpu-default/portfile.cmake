if(NOT VCPKG_TARGET_IS_LINUX)
    message(FATAL_ERROR "onnxruntime-gpu-default overlay port currently supports Linux only.")
endif()

set(ORT_VERSION "v1.10.0")
set(ORT_SOURCE_DIR "${CURRENT_BUILDTREES_DIR}/src/onnxruntime-${ORT_VERSION}")

vcpkg_execute_required_process(
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${ORT_SOURCE_DIR}"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
    LOGNAME "cleanup-source"
)

vcpkg_execute_required_process(
    COMMAND git clone --depth 1 --branch ${ORT_VERSION} https://github.com/microsoft/onnxruntime.git "${ORT_SOURCE_DIR}"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
    LOGNAME "clone-source"
)

set(ORT_BUILD_ARGS
    --config Release
    --update
    --build
    --skip_tests
    --parallel
    --build_shared_lib
)

if(EXISTS "/usr/local/cuda")
    list(APPEND ORT_BUILD_ARGS --use_cuda --cuda_home /usr/local/cuda)
    if(EXISTS "/usr/include/cudnn.h" OR EXISTS "/usr/include/cudnn_version.h")
        list(APPEND ORT_BUILD_ARGS --cudnn_home /usr)
    endif()
endif()

if(EXISTS "/usr/include/NvInfer.h")
    list(APPEND ORT_BUILD_ARGS --use_tensorrt --tensorrt_home /usr)
endif()

vcpkg_execute_required_process(
    COMMAND /usr/bin/env bash ./build.sh ${ORT_BUILD_ARGS}
    WORKING_DIRECTORY "${ORT_SOURCE_DIR}"
    LOGNAME "build-${TARGET_TRIPLET}-rel"
)

set(ORT_LIB_DIR "${ORT_SOURCE_DIR}/build/Linux/Release")
if(NOT EXISTS "${ORT_LIB_DIR}")
    message(FATAL_ERROR "onnxruntime build output not found at ${ORT_LIB_DIR}")
endif()

file(GLOB ORT_SHARED_LIBS
    "${ORT_LIB_DIR}/libonnxruntime.so"
    "${ORT_LIB_DIR}/libonnxruntime.so.*"
    "${ORT_LIB_DIR}/libonnxruntime_providers_*.so"
)
if(ORT_SHARED_LIBS STREQUAL "")
    message(FATAL_ERROR "No onnxruntime shared libraries produced in ${ORT_LIB_DIR}")
endif()

file(INSTALL
    DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
    TYPE FILE
    FILES ${ORT_SHARED_LIBS}
)

file(INSTALL
    DESTINATION "${CURRENT_PACKAGES_DIR}/include/onnxruntime"
    TYPE DIRECTORY
    FILES "${ORT_SOURCE_DIR}/include/onnxruntime/"
)

file(INSTALL
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    TYPE FILE
    FILES "${ORT_SOURCE_DIR}/LICENSE"
    RENAME copyright
)
