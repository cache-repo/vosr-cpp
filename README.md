# VOSR MNN C++ 推理

基于 `vosr-python-mnn` 的 MNN Python 推理实现，用 C++ 重写同一条推理链路。
模型文件复用 `vosr-python-mnn/` 下已转换好的 `.mnn`，支持 `vosr-0.5b-os` 和 `vosr2` 两套，无需重新导出。

## 与 Python 版本的关系

| 项目 | Python 版 | C++ 版 |
|---|---|---|
| 模型 | `vosr-0.5b-os` / `vosr2` 两套 | 同左，按 `--model` 切换 |
| MNN API | `MNN.nn` / `MNN.expr` Python 绑定 | `MNN::Express::Module` C++ API |
| 输入 / 输出 | Pillow 图片 | stb_image / stb_image_write |
| 采样 / 分块 / 色彩对齐 | numpy | 纯 C++（std::vector + stb_image_resize） |

四个子模型阶段与 Python 版一致：

| 子模型 | 作用 | 输入输出 |
|---|---|---|
| `vae_encoder.mnn` | VAE 编码（归一化折入） | `[1,3,H,W]` -> `[1,C,H/8,W/8]` |
| `dinov2.mnn` | DINOv2 指定层特征 | `[1,3,448,448]` -> `[1,N,C]` |
| `dit.mnn` | DiT 单步速度场 | `[x(2C), zfea, t(, r)]` -> `velocity` |
| `vae_decoder.mnn` | VAE 解码 | `[1,C,h,w]` -> `[1,3,h*8,w*8]` |

`vosr-0.5b-os`：sd2 / 4ch / DINOv2-B / DiT 4 输入。
`vosr2`：qwen / 16ch / DINOv2-L / DiT 3 输入（无 r）。
vosr2 的 `dit.mnn` 超过 2 GB，MNN 拆出的 `dit.mnn.weight` 必须与 `.mnn` 放在同一目录。

采样循环、VAE 分块高斯混合、DINOv2 预处理、t tile 混合、色彩对齐与 Python 版一致。

## 构建

前置条件：CMake 3.16+、C++14 编译器、MNN 源码（仓库根的 `MNN` submodule）。

```bash
# 方式 1：使用仓库内 submodule 自动构建 MNN（推荐）
cd vosr-mnn-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

如果已经自行编译过 MNN，可直接指定：

```bash
cmake -S . -B build \
  -DMNN_INCLUDE_DIR=/path/to/MNN/include \
  -DMNN_LIBRARY=/path/to/libMNN.a
cmake --build build -j4
```

产物：`vosr-mnn-cpp/build/vosr-mnn-cpp`。

## 推理

```bash
# 单图 4x 放大（默认 tile-size 512 / adain / 4 线程）
vosr-mnn-cpp/build/vosr-mnn-cpp \
  -i vosr-python-mnn/assets/benchmark.png \
  -o output.png \
  -u 4 --tile-size 512 --align adain --threads 4

# vosr2
vosr-mnn-cpp/build/vosr-mnn-cpp \
  -i vosr-python-mnn/assets/benchmark.png \
  -o output-vosr2.png \
  --model vosr2 -u 4 --tile-size 512 --threads 4
```

参数与 Python 版对齐：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-i, --input` | 必填 | 输入 PNG / JPG / BMP / TIFF 图片 |
| `-o, --output` | 必填 | 输出 PNG 路径 |
| `--model` | `vosr-0.5b-os` | 模型：`vosr-0.5b-os` / `vosr2` |
| `--model-dir` | 按 `--model` 选择 | MNN 模型目录（显式指定时覆盖默认值） |
| `-u, --upscale` | 4 | 放大倍数 |
| `--steps` | 1 | 采样步数 |
| `--seed` | 42 | 随机种子 |
| `--tile-size` | 512 | DiT / VAE 分块尺寸，需与 `dit.mnn` 的 64x64 latent 匹配 |
| `--tile-overlap` | 32 | 分块重叠像素 |
| `--align` | adain | `adain` / `wavelet` / `nofix` |
| `--threads` | 4 | MNN CPU 线程数 |
| `--backend` | cpu | `cpu` / `opencl`（opencl 不可用时回退 cpu） |
| `--opencl-memory` | buffer | `buffer` / `image`（OpenCL 内存模式，见下） |

OpenCL 后端（需 MNN 带 OpenCL 编译，且目标机器有可用 ICD/设备）：

```bash
# 原生构建默认带 OpenCL（CMAKE_CROSSCOMPILING 时自动关闭）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

build/vosr-mnn-cpp \
  -i ../vosr-python-mnn/assets/benchmark.png -o output-opencl.png \
  --model-dir ../vosr-python-mnn/models/mnn-vosr_0.5B_os_fp16 \
  -u 4 --tile-size 512 --backend opencl --threads 4
```

Windows / Wine 交叉编译同样可带 OpenCL（需显式开启，交叉编译默认关闭）：

```bash
cmake -S . -B build-wine \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/mingw-wine-toolchain.cmake" \
  -DVOSR_MNN_OPENCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-wine -j4
cp /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll build-wine/  # MinGW posix 线程 DLL
```

MNN OpenCL 走运行时加载（Windows 为 `LoadLibraryA("OpenCL.dll")`，Linux 为
`dlopen("libOpenCL.so")`），无需链接 OpenCL 开发包。Wine 自带 `opencl.dll` 桥接到
宿主 ICD；没有 ICD 或 GPU 设备时 MNN 打印回退日志并改用 CPU，结果与 `--backend cpu` 一致
（MNN 只枚举 `CL_DEVICE_TYPE_GPU`，POCL 的 CPU 设备也会回退）。
显式关闭 OpenCL 可加 `-DVOSR_MNN_OPENCL=OFF`。

`--opencl-memory buffer`（默认）把 MNN 的 gpuMode 设为 `MNN_GPU_MEMORY_BUFFER`，用
`cl::Buffer` 而非 `cl::Image2D` 存放张量。这绕开了 `CL_DEVICE_IMAGE2D_MAX_HEIGHT`
限制——IMAGE 模式下 DiT 的大权重会被映射成 `4x37632` 的 Image2D，在多数独显上
（上限 16384/32768）分配失败并报 `CL_INVALID_IMAGE_SIZE (code:-40)`，且该 STATIC
权重分配早于 MNN 的自动尺寸回退检查。若 GPU 支持更大 Image2D、或 IMAGE 模式更快，
可传 `--opencl-memory image` 切回。性能因 GPU 而异，可两种模式对比后选择。

注意：`dit.mnn` 的 RoPE 固定为 64x64 latent，因此 `--tile-size` 必须是 512。
其他值需要重新导出 DiT 模型。

## 限制

- 当前支持 `vosr-0.5b-os`（sd2 VAE + DINOv2-B）和 `vosr2`（qwen VAE + DINOv2-L）两套 MNN 子模型。
- 仅实现单张图片推理，目录批量处理可通过 shell 循环调用。
- 色彩 `wavelet` 实现用 box blur 近似 Python 的 PIL GaussianBlur，输出与 Python 版可能有细微差异。
