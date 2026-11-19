#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vosr {

struct Options {
    std::string model = "vosr-0.5b-os";
    std::string model_dir;
    int latent_channels = 4;
    bool dit_has_r = true;
    int threads = 4;
    std::string backend = "cpu"; // cpu | opencl
    std::string opencl_memory = "buffer"; // buffer | image（buffer 不受 Image2D 尺寸上限限制）
    int upscale = 4;
    int steps = 1;
    int seed = 42;
    int tile_size = 512;
    int tile_overlap = 32;
    std::string align = "adain"; // wavelet | adain | nofix
};

struct Image {
    int w = 0, h = 0, c = 3;
    std::vector<uint8_t> data;
};

// 主推理接口：输入 RGB 图像 -> RGB 超分图像，并写入 output_path。
Image infer(const Options& opt, const Image& input, const std::string& output_path);

// 文件 I/O（stb）
bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img);

} // namespace vosr
