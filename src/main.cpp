#include "vosr.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(const char* prog) {
    std::cout << "用法: " << prog << " -i <输入图片> -o <输出图片> [选项]\n"
              << "选项:\n"
              << "  --model NAME      模型：vosr-0.5b-os|vosr2（默认 vosr-0.5b-os）\n"
              << "  --model-dir DIR   MNN 模型目录（默认按 --model 选择）\n"
              << "  -u N              放大倍数（默认 4）\n"
              << "  --steps N         采样步数（默认 1）\n"
              << "  --seed N          随机种子（默认 42）\n"
              << "  --tile-size N     DiT 分块像素尺寸（默认 512，需与 dit.mnn 匹配）\n"
              << "  --tile-overlap N  分块重叠像素（默认 32）\n"
              << "  --align NAME      色彩对齐：adain|wavelet|nofix（默认 adain）\n"
              << "  --threads N       MNN 线程数（默认 4）\n"
              << "  --backend NAME    推理后端：cpu|opencl（默认 cpu；opencl 不可用时回退 cpu）\n"
              << "  --opencl-memory   OpenCL 内存模式：buffer|image（默认 buffer，避免 Image2D 尺寸限制）\n";
}

std::string arg(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("缺少参数值: ") + argv[i]);
    return argv[++i];
}

int parseInt(const std::string& s) { return std::stoi(s); }

} // namespace

int main(int argc, char** argv) {
    try {
        vosr::Options opt;
        std::string input, output;
        opt.model_dir.clear();
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
            else if (a == "-i" || a == "--input") input = arg(argc, argv, i);
            else if (a == "-o" || a == "--output") output = arg(argc, argv, i);
            else if (a == "--model") opt.model = arg(argc, argv, i);
            else if (a == "--model-dir") opt.model_dir = arg(argc, argv, i);
            else if (a == "-u" || a == "--upscale") opt.upscale = parseInt(arg(argc, argv, i));
            else if (a == "--steps") opt.steps = parseInt(arg(argc, argv, i));
            else if (a == "--seed") opt.seed = parseInt(arg(argc, argv, i));
            else if (a == "--tile-size") opt.tile_size = parseInt(arg(argc, argv, i));
            else if (a == "--tile-overlap") opt.tile_overlap = parseInt(arg(argc, argv, i));
            else if (a == "--align") opt.align = arg(argc, argv, i);
            else if (a == "--threads") opt.threads = parseInt(arg(argc, argv, i));
            else if (a == "--backend") opt.backend = arg(argc, argv, i);
            else if (a == "--opencl-memory") opt.opencl_memory = arg(argc, argv, i);
            else throw std::runtime_error("未知参数: " + a);
        }
        if (input.empty() || output.empty()) {
            usage(argv[0]);
            return 1;
        }
        // 按 --model 选择默认目录与 latent 通道数；显式 --model-dir 优先。
        if (opt.model == "vosr2") {
            opt.latent_channels = 16;
            opt.dit_has_r = false;
            if (opt.model_dir.empty()) opt.model_dir = "vosr-python-mnn/models/mnn-vosr2";
        } else if (opt.model == "vosr-0.5b-os") {
            opt.latent_channels = 4;
            opt.dit_has_r = true;
            if (opt.model_dir.empty()) opt.model_dir = "vosr-python-mnn/models/mnn";
        } else {
            throw std::runtime_error("未知 --model: " + opt.model);
        }
        // 默认模型目录是相对仓库根的路径；允许在 vosr-mnn-cpp 子目录运行。
        // 显式 --model-dir 不做路径改写。
        if (opt.model_dir == "vosr-python-mnn/models/mnn" ||
            opt.model_dir == "vosr-python-mnn/models/mnn-vosr2") {
            std::ifstream probe(opt.model_dir + "/vae_encoder.mnn", std::ios::binary);
            if (!probe) opt.model_dir = "../" + opt.model_dir;
        }
        vosr::Image img;
        if (!vosr::load_image(input, img))
            throw std::runtime_error("读取输入图片失败: " + input);
        vosr::infer(opt, img, output);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[vosr-cpp] ERROR: " << e.what() << std::endl;
        return 1;
    }
}
