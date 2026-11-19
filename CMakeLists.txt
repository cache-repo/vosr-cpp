cmake_minimum_required(VERSION 3.16)
project(vosr-mnn-cpp CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(MNN_SOURCE_DIR "${REPO_ROOT}/MNN" CACHE PATH "MNN 源码目录")

# MNN 头文件与库：优先从 submodule 就地构建；可用 MNN_INCLUDE_DIR / MNN_LIBRARY 覆盖。
set(MNN_INCLUDE_DIR "" CACHE PATH "MNN include 目录")
set(MNN_LIBRARY "" CACHE FILEPATH "MNN 静态库 libMNN.a")
set(MNN_TRAIN_LIBRARY "" CACHE FILEPATH "MNNTrain 静态库")

# OpenCL 后端：默认开启；交叉编译（如 MinGW/Wine）默认关闭，
# 但可显式 -DVOSR_MNN_OPENCL=ON 打开（MNN OpenCL 用运行时 LoadLibraryA/dlopen
# 加载 OpenCL.dll/libOpenCL，无 ICD 时自动回退 CPU）。
if(CMAKE_CROSSCOMPILING)
  set(_vosr_opencl_default OFF)
else()
  set(_vosr_opencl_default ON)
endif()
option(VOSR_MNN_OPENCL "构建带 OpenCL 后端的 MNN（支持 --backend opencl）" ${_vosr_opencl_default})

if(MNN_INCLUDE_DIR AND MNN_LIBRARY)
  set(MNN_INC "${MNN_INCLUDE_DIR}")
  set(MNN_LIB "${MNN_LIBRARY}")
  if(MNN_TRAIN_LIBRARY)
    set(MNN_TRAIN_LIB "${MNN_TRAIN_LIBRARY}")
  endif()
elseif(EXISTS "${MNN_SOURCE_DIR}/CMakeLists.txt")
  set(MNN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mnn-build" CACHE PATH "MNN 构建目录")
  if(NOT EXISTS "${MNN_BINARY_DIR}/libMNN.a")
    # MNN OpenCL 源码用 <Windows.h>（大写 W）；MinGW 头文件是小写 windows.h，
    # 在大小写敏感的 Linux 上找不到。生成一个仅含转发的 shim 目录给内层 MNN。
    if(CMAKE_CROSSCOMPILING AND VOSR_MNN_OPENCL)
      set(_win_shim "${CMAKE_CURRENT_BINARY_DIR}/win32-case-shim")
      file(MAKE_DIRECTORY "${_win_shim}")
      file(WRITE "${_win_shim}/Windows.h" "#pragma once\n#include <windows.h>\n")
    endif()
    # 交叉编译时把工具链透传给内层 MNN 配置，避免 MNN 以 Linux 方式编译 Windows 目标。
    set(_mnn_cmake_args -DCMAKE_BUILD_TYPE=Release)
    if(_win_shim)
      list(APPEND _mnn_cmake_args "-DCMAKE_CXX_FLAGS=-I${_win_shim}")
    endif()
    if(CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_toolchain_abs "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
        list(APPEND _mnn_cmake_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain_abs}")
    endif()
    if(CMAKE_SYSTEM_NAME)
        list(APPEND _mnn_cmake_args "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")
        # 必须同时透传处理器架构：只给 SYSTEM_NAME 会让内层 CMake 误判为交叉编译，
        # CMAKE_SYSTEM_PROCESSOR 变空，MNN 的 x86 AVX2/SSE 优化整段被跳过（性能差数倍）。
        if(CMAKE_SYSTEM_PROCESSOR)
            list(APPEND _mnn_cmake_args "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
        endif()
    endif()
    foreach(_cc CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_ASM_COMPILER CMAKE_RC_COMPILER)
        if(${_cc})
            list(APPEND _mnn_cmake_args "-D${_cc}=${${_cc}}")
        endif()
    endforeach()
    execute_process(COMMAND ${CMAKE_COMMAND} -S "${MNN_SOURCE_DIR}" -B "${MNN_BINARY_DIR}" ${_mnn_cmake_args}
        -DMNN_BUILD_SHARED_LIBS=OFF
        -DMNN_SEP_BUILD=OFF
        -DMNN_USE_THREAD_POOL=ON
        -DMNN_OPENMP=OFF
        -DMNN_BUILD_TRAIN=ON
        -DMNN_BUILD_CONVERTER=OFF
        -DMNN_BUILD_DEMO=OFF
        -DMNN_BUILD_TOOLS=OFF
        -DMNN_BUILD_TEST=OFF
        -DMNN_BUILD_BENCHMARK=OFF
        -DMNN_BUILD_OPENCV=OFF
        -DMNN_IMGCODECS=OFF
        -DMNN_LOW_MEMORY=ON
        -DMNN_USE_SSE=ON
        -DMNN_AVX2=ON
        -DMNN_OPENCL=${VOSR_MNN_OPENCL}
      RESULT_VARIABLE _r)
    if(NOT _r EQUAL 0)
      message(FATAL_ERROR "MNN CMake 配置失败")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build "${MNN_BINARY_DIR}" --target MNN MNNTrain -j4
      RESULT_VARIABLE _r)
    if(NOT _r EQUAL 0)
      message(FATAL_ERROR "MNN 构建失败")
    endif()
  endif()
  set(MNN_INC "${MNN_SOURCE_DIR}/include")
  set(MNN_LIB "${MNN_BINARY_DIR}/libMNN.a")
  file(GLOB _train_lib "${MNN_BINARY_DIR}/tools/train/libMNNTrain.a")
  if(_train_lib)
    set(MNN_TRAIN_LIB "${_train_lib}")
  endif()
else()
  message(FATAL_ERROR "未找到 MNN：设置 MNN_INCLUDE_DIR/MNN_LIBRARY，或初始化仓库根 MNN submodule")
endif()

add_executable(vosr-mnn-cpp src/main.cpp src/vosr.cpp)
target_include_directories(vosr-mnn-cpp PRIVATE src third_party "${MNN_INC}")
target_link_libraries(vosr-mnn-cpp PRIVATE "${MNN_LIB}")
if(MNN_TRAIN_LIB)
  target_link_libraries(vosr-mnn-cpp PRIVATE "${MNN_TRAIN_LIB}")
endif()
if(MSVC)
  target_compile_options(vosr-mnn-cpp PRIVATE /O2)
else()
  # MinGW 运行时 DLL 不在 Wine prefix 中，静态链接 libgcc/libstdc++ 便于直接 wine 运行。
  target_compile_options(vosr-mnn-cpp PRIVATE -O3 -pthread)
  target_link_options(vosr-mnn-cpp PRIVATE -static-libgcc -static-libstdc++)
  find_package(Threads REQUIRED)
  target_link_libraries(vosr-mnn-cpp PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

install(TARGETS vosr-mnn-cpp RUNTIME DESTINATION bin)
