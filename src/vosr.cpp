#include "vosr.hpp"

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TIFF
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

namespace vosr {
namespace F = MNN::Express;

namespace {

constexpr int AE_FACTOR = 8;
constexpr int DINO_SIDE = 448;
constexpr int DIT_LATENT = 64;
const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

void log(const std::string& s) { std::cout << "[vosr-cpp] " << s << std::endl; }

// ---- Tensor: [N,C,H,W] float32, NCHW 连续 ----
struct Tensor {
    std::vector<int> shape; // {N,C,H,W}
    std::vector<float> data;
    int n() const { return shape[0]; }
    int c() const { return shape[1]; }
    int h() const { return shape[2]; }
    int w() const { return shape[3]; }
    size_t size() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

Tensor zeros(const std::vector<int>& shape) {
    Tensor t;
    t.shape = shape;
    size_t n = 1;
    for (int x : shape) n *= static_cast<size_t>(x);
    t.data.assign(n, 0.0f);
    return t;
}

// ---- MNN Module 封装：同一 shape 的 VARP/placeholder 复用，等价于 Python MNNModule ----
class MNNModel {
public:
    MNNModel(const std::string& path, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs,
             const std::shared_ptr<MNN::Express::Executor::RuntimeManager>& rt)
        : names_(inputs) {
        MNN::Express::Module::Config cfg;
        cfg.dynamic = false;
        cfg.shapeMutable = true;
        module_.reset(MNN::Express::Module::load(inputs, outputs, path.c_str(), rt, &cfg));
        if (!module_) throw std::runtime_error("加载 MNN 模型失败: " + path);
    }

    Tensor forward(const std::vector<Tensor>& xs) {
        std::vector<int> key;
        for (const auto& x : xs) key.insert(key.end(), x.shape.begin(), x.shape.end());
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            std::vector<F::VARP> vars;
            for (const auto& x : xs) {
                vars.push_back(F::_Input(x.shape, F::NCHW, halide_type_of<float>()));
            }
            it = cache_.emplace(key, std::move(vars)).first;
        }
        for (size_t i = 0; i < xs.size(); ++i) {
            std::memcpy(it->second[i]->writeMap<float>(), xs[i].ptr(),
                        xs[i].size() * sizeof(float));
        }
        auto outs = module_->onForward(it->second);
        if (outs.empty()) throw std::runtime_error("MNN 模型无输出");
        auto info = outs[0]->getInfo();
        Tensor out;
        out.shape = info->dim;
        size_t out_n = 1;
        for (int d : out.shape) out_n *= static_cast<size_t>(d);
        out.data.assign(out_n, 0.0f);
        std::memcpy(out.ptr(), outs[0]->readMap<float>(), out_n * sizeof(float));
        return out;
    }

private:
    std::vector<std::string> names_;
    std::shared_ptr<MNN::Express::Module> module_;
    std::map<std::vector<int>, std::vector<F::VARP>> cache_;
};

// ---- 通用工具 ----
Image resize_bicubic(const Image& src, int nw, int nh) {
    Image dst;
    dst.w = nw; dst.h = nh; dst.c = src.c;
    dst.data.resize(static_cast<size_t>(nw) * nh * src.c);
    stbir_resize_uint8_generic(src.data.data(), src.w, src.h, src.w * src.c,
                       dst.data.data(), nw, nh, nw * dst.c, src.c, -1, 0,
                       STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, nullptr);
    return dst;
}

Tensor image_to_tensor(const Image& img) { // [-1,1]
    Tensor t = zeros({1, 3, img.h, img.w});
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                t.data[(c * img.h + y) * img.w + x] =
                    img.data[(y * img.w + x) * 3 + c] / 255.0f * 2.0f - 1.0f;
            }
        }
    }
    return t;
}

Image tensor_to_image(const Tensor& t) {
    Image img;
    img.c = 3; img.h = t.h(); img.w = t.w();
    img.data.resize(static_cast<size_t>(img.w) * img.h * 3);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = t.data[(c * img.h + y) * img.w + x] * 0.5f + 0.5f;
                img.data[(y * img.w + x) * 3 + c] =
                    static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f);
            }
        }
    }
    return img;
}

Tensor pad_tile(const Tensor& x, int size) {
    if (x.h() == size && x.w() == size) return x;
    Tensor out = zeros({x.n(), x.c(), size, size});
    const int h = x.h(), w = x.w();
    for (int n = 0; n < x.n(); ++n) {
        for (int c = 0; c < x.c(); ++c) {
            for (int y = 0; y < h; ++y) {
                std::memcpy(out.ptr() + ((n * x.c() + c) * size + y) * size,
                            x.ptr() + ((n * x.c() + c) * h + y) * w, w * sizeof(float));
                for (int xx = w; xx < size; ++xx)
                    out.data[((n * x.c() + c) * size + y) * size + xx] =
                        x.data[((n * x.c() + c) * h + y) * w + w - 1];
            }
            for (int y = h; y < size; ++y) {
                std::memcpy(out.ptr() + ((n * x.c() + c) * size + y) * size,
                            out.ptr() + ((n * x.c() + c) * size + (h - 1)) * size,
                            size * sizeof(float));
            }
        }
    }
    return out;
}

Tensor crop(const Tensor& x, int oh, int ow) {
    Tensor out = zeros({x.n(), x.c(), oh, ow});
    for (int n = 0; n < x.n(); ++n)
        for (int c = 0; c < x.c(); ++c)
            for (int y = 0; y < oh; ++y)
                std::memcpy(out.ptr() + ((n * x.c() + c) * oh + y) * ow,
                            x.ptr() + ((n * x.c() + c) * x.h() + y) * x.w(),
                            ow * sizeof(float));
    return out;
}

std::vector<int> tile_grid(int length, int tile, int overlap) {
    int stride = std::max(tile - overlap, 1);
    if (length <= tile) return {0};
    std::vector<int> pos;
    for (int p = 0; p <= length - tile; p += stride) pos.push_back(p);
    if (pos.back() + tile < length) pos.push_back(length - tile);
    std::sort(pos.begin(), pos.end());
    pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
    return pos;
}

std::vector<float> gaussian_weights(int size, float var = 0.01f) {
    float mid = (size - 1) / 2.0f;
    std::vector<float> w(size);
    for (int i = 0; i < size; ++i) {
        float d = (i - mid) / size;
        w[i] = std::exp(-(d * d) / (2.0f * var));
    }
    std::vector<float> out(static_cast<size_t>(size) * size);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            out[y * size + x] = w[y] * w[x];
    return out;
}

Tensor concat_channels(const Tensor& a, const Tensor& b) {
    Tensor out = zeros({a.n(), a.c() + b.c(), a.h(), a.w()});
    size_t asz = static_cast<size_t>(a.n()) * a.c() * a.h() * a.w();
    std::memcpy(out.ptr(), a.ptr(), asz * sizeof(float));
    std::memcpy(out.ptr() + asz, b.ptr(), b.size() * sizeof(float));
    return out;
}

// DINOv2 预处理：与 Python dino_preprocess 对齐。
// raw = (0.5*lq + 0.5)*255 -> bicubic resize 448 -> /255 -> 减均值除方差
Tensor dino_preprocess(const Tensor& lq) {
    Tensor out = zeros({1, 3, DINO_SIDE, DINO_SIDE});
    Image plane;
    plane.c = 1;
    plane.w = lq.w(); plane.h = lq.h();
    plane.data.resize(static_cast<size_t>(plane.w) * plane.h);
    Image resized;
    resized.c = 1; resized.w = DINO_SIDE; resized.h = DINO_SIDE;
    resized.data.resize(static_cast<size_t>(DINO_SIDE) * DINO_SIDE);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < plane.h; ++y)
            for (int x = 0; x < plane.w; ++x) {
                float v = (0.5f * lq.data[(c * lq.h() + y) * lq.w() + x] + 0.5f) * 255.0f;
                plane.data[y * plane.w + x] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, v)));
            }
        stbir_resize_uint8_generic(plane.data.data(), plane.w, plane.h, plane.w,
                           resized.data.data(), DINO_SIDE, DINO_SIDE, DINO_SIDE, 1, -1, 0,
                           STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, nullptr);
        for (int i = 0; i < DINO_SIDE * DINO_SIDE; ++i) {
            float v = resized.data[i] / 255.0f;
            out.data[(c * DINO_SIDE * DINO_SIDE) + i] =
                (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    return out;
}

// ---- 色彩对齐 ----
void color_align_adain(Image& target, const Image& source) {
    Image src = resize_bicubic(source, target.w, target.h);
    float tm[3] = {0}, sm[3] = {0}, ts[3] = {0}, ss[3] = {0};
    const size_t n = static_cast<size_t>(target.w) * target.h;
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            tm[c] += target.data[i * 3 + c] / 255.0f;
            sm[c] += src.data[i * 3 + c] / 255.0f;
        }
    }
    for (int c = 0; c < 3; ++c) { tm[c] /= n; sm[c] /= n; }
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float tv = target.data[i * 3 + c] / 255.0f - tm[c];
            float sv = src.data[i * 3 + c] / 255.0f - sm[c];
            ts[c] += tv * tv; ss[c] += sv * sv;
        }
    }
    for (int c = 0; c < 3; ++c) {
        ts[c] = std::sqrt(ts[c] / n) + 1e-5f;
        ss[c] = std::sqrt(ss[c] / n) + 1e-5f;
    }
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (target.data[i * 3 + c] / 255.0f - tm[c]) / ts[c] * ss[c] + sm[c];
            target.data[i * 3 + c] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f);
        }
    }
}

// 简化 wavelet：source 低频 + target 高频，用 3x3 box blur 近似 GaussianBlur(5)。
void color_align_wavelet(Image& target, const Image& source) {
    Image src = resize_bicubic(source, target.w, target.h);
    auto blur = [](const Image& in) {
        Image out = in;
        const int r = 2;
        for (int y = 0; y < in.h; ++y) {
            for (int x = 0; x < in.w; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0; int cnt = 0;
                    for (int dy = -r; dy <= r; ++dy) {
                        int yy = std::max(0, std::min(in.h - 1, y + dy));
                        for (int dx = -r; dx <= r; ++dx) {
                            int xx = std::max(0, std::min(in.w - 1, x + dx));
                            sum += in.data[(yy * in.w + xx) * 3 + c]; ++cnt;
                        }
                    }
                    out.data[(y * in.w + x) * 3 + c] = static_cast<uint8_t>(sum / cnt);
                }
            }
        }
        return out;
    };
    Image tl = blur(target), sl = blur(src);
    for (size_t i = 0; i < target.data.size(); ++i) {
        int v = static_cast<int>(sl.data[i]) + static_cast<int>(target.data[i]) - static_cast<int>(tl.data[i]);
        target.data[i] = static_cast<uint8_t>(std::max(0, std::min(255, v)));
    }
}

// ---- Engine ----
class Engine {
public:
    Engine(const Options& opt, const std::shared_ptr<MNN::Express::Executor::RuntimeManager>& rt)
        : opt_(opt),
          enc_(path("vae_encoder.mnn"), {"image"}, {"latent"}, rt),
          dino_(path("dinov2.mnn"), {"image"}, {"feat"}, rt),
          dit_(path("dit.mnn"),
               opt.dit_has_r ? std::vector<std::string>{"x", "zfea", "t", "r"}
                             : std::vector<std::string>{"x", "zfea", "t"},
               {"velocity"}, rt),
          dec_(path("vae_decoder.mnn"), {"latent"}, {"image"}, rt) {
        log("4 个 MNN 子模型加载完成");
    }

    Image run(const Image& input) {
        const int tw = input.w * opt_.upscale, th = input.h * opt_.upscale;
        log("输入 " + std::to_string(input.w) + "x" + std::to_string(input.h) +
            " -> 输出 " + std::to_string(tw) + "x" + std::to_string(th));

        auto bicubic = resize_bicubic(input, tw, th);
        auto lq = image_to_tensor(bicubic);
        auto latent = tiled_encode(lq);
        log("VAE encode: " + std::to_string(latent.h()) + "x" + std::to_string(latent.w()));

        bool tiled = opt_.tile_size > 0 && (latent.h() > DIT_LATENT || latent.w() > DIT_LATENT);
        Tensor zfea;
        if (!tiled) {
            zfea = dino_.forward({dino_preprocess(lq)});
            log("DINOv2 全局特征: " + std::to_string(zfea.shape[1]) + "x" +
                std::to_string(zfea.shape[2]));
        } else {
            log("DINOv2 全局特征: 分块采样不需要，已跳过");
        }

        auto sr_latent = sample(latent, lq, zfea, tiled);
        auto sr = tiled_decode(sr_latent);
        // qwen VAE 解码按 floor(H/8)*8，可能比目标小 1~7 像素；
        // 保持 VAE 实际输出尺寸，与上游一致。
        auto img = tensor_to_image(crop(sr, std::min(sr.h(), th), std::min(sr.w(), tw)));

        if (opt_.align == "adain") color_align_adain(img, bicubic);
        else if (opt_.align == "wavelet") color_align_wavelet(img, bicubic);
        return img;
    }

private:
    std::string path(const std::string& name) const {
        return opt_.model_dir + "/" + name;
    }

    Tensor dit_forward(const Tensor& x, const Tensor& fea, float t, float r) {
        Tensor tt = zeros({1}); tt.data[0] = t;
        Tensor rt = zeros({1}); rt.data[0] = r;
        if (opt_.dit_has_r) return dit_.forward({x, fea, tt, rt});
        return dit_.forward({x, fea, tt});
    }

    Tensor get_dino_features(const Tensor& lq_tile) {
        return dino_.forward({dino_preprocess(lq_tile)});
    }

    Tensor sample(const Tensor& lq_latent, const Tensor& lq, const Tensor& zfea, bool tiled) {
        std::mt19937 rng(static_cast<uint32_t>(opt_.seed));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        Tensor z = zeros(lq_latent.shape);
        for (auto& v : z.data) v = nd(rng);
        const int h = lq_latent.h(), w = lq_latent.w();
        std::vector<float> tseq(opt_.steps + 1);
        for (int i = 0; i <= opt_.steps; ++i)
            tseq[i] = 1.0f - static_cast<float>(i) / opt_.steps;

        if (!tiled) {
            Tensor tile_lq = pad_tile(lq_latent, DIT_LATENT);
            Tensor tile_z = pad_tile(z, DIT_LATENT);
            log("latent " + std::to_string(h) + "x" + std::to_string(w) +
                "，单块推理（pad 到 64x64，" + std::to_string(opt_.steps) + " 步）");
            for (int i = 0; i < opt_.steps; ++i) {
                float t = tseq[i], r = tseq[i + 1];
                auto x = concat_channels(tile_lq, tile_z);
                auto v = dit_forward(x, zfea, t, r);
                float dt = t - r;
                for (size_t j = 0; j < tile_z.size(); ++j) tile_z.data[j] -= dt * v.data[j];
            }
            return crop(tile_z, h, w);
        }

        int lt = opt_.tile_size / AE_FACTOR;
        lt = (lt / 2) * 2;
        if (lt != DIT_LATENT)
            throw std::runtime_error("--tile-size 需要为 512 才能匹配 dit.mnn 的 64x64 latent");
        int ov = std::min(std::max(opt_.tile_overlap / AE_FACTOR, lt / 8), lt - 1);
        auto hs = tile_grid(h, lt, ov), ws = tile_grid(w, lt, ov);
        auto g = gaussian_weights(lt);
        log("latent " + std::to_string(h) + "x" + std::to_string(w) +
            ", tile=" + std::to_string(lt) + ", overlap=" + std::to_string(ov) +
            ", grid=" + std::to_string(hs.size()) + "x" + std::to_string(ws.size()));

        for (int i = 0; i < opt_.steps; ++i) {
            float t = tseq[i], r = tseq[i + 1];
            Tensor vacc = zeros(z.shape), wacc = zeros(z.shape);

            // 逐 tile 重提 DINO 特征，与 Python tiled_latent_inference 一致
            struct Rec { int hi, wi, he, we, th, tw; Tensor v; };
            std::vector<Rec> preds;
            for (int hi : hs) {
                for (int wi : ws) {
                    int ps = hi * AE_FACTOR, pw = wi * AE_FACTOR;
                    int pe = std::min((hi + lt) * AE_FACTOR, lq.h());
                    int pwe = std::min((wi + lt) * AE_FACTOR, lq.w());
                    Tensor crop_lq = zeros({1, 3, pe - ps, pwe - pw});
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < pe - ps; ++y)
                            std::memcpy(crop_lq.ptr() + (c * (pe - ps) + y) * (pwe - pw),
                                        lq.ptr() + (c * lq.h() + ps + y) * lq.w() + pw,
                                        (pwe - pw) * sizeof(float));
                    auto fea = get_dino_features(crop_lq);

                    int he = std::min(hi + lt, h), we = std::min(wi + lt, w);
                    int th = he - hi, tw = we - wi;
                    auto lq_tile = pad_tile(lq_latent_region(lq_latent, hi, he, wi, we), lt);
                    auto z_tile = pad_tile(z_region(z, hi, he, wi, we), lt);
                    auto x = concat_channels(lq_tile, z_tile);
                    auto v = dit_forward(x, fea, t, r);
                    preds.push_back({hi, wi, he, we, th, tw, v});
                }
            }
            for (const auto& p : preds) {
                for (int c = 0; c < opt_.latent_channels; ++c) {
                    for (int y = 0; y < p.th; ++y) {
                        for (int x = 0; x < p.tw; ++x) {
                            float gw = g[y * lt + x];
                            size_t idx = (static_cast<size_t>(c) * h + p.hi + y) * w + p.wi + x;
                            vacc.data[idx] += p.v.data[(c * lt + y) * lt + x] * gw;
                            wacc.data[idx] += gw;
                        }
                    }
                }
            }
            float dt = t - r;
            for (size_t j = 0; j < z.size(); ++j)
                z.data[j] -= dt * (vacc.data[j] / std::max(wacc.data[j], 1e-12f));
        }
        return z;
    }

    // 轻量区域视图：返回连续内存的子区域副本（只用于 DIT tile）
    static Tensor lq_latent_region(const Tensor& x, int y0, int y1, int x0, int x1) {
        return region(x, y0, y1, x0, x1);
    }
    static Tensor z_region(const Tensor& x, int y0, int y1, int x0, int x1) {
        return region(x, y0, y1, x0, x1);
    }
    static Tensor region(const Tensor& x, int y0, int y1, int x0, int x1) {
        int oh = y1 - y0, ow = x1 - x0;
        Tensor out = zeros({x.n(), x.c(), oh, ow});
        for (int n = 0; n < x.n(); ++n)
            for (int c = 0; c < x.c(); ++c)
                for (int y = 0; y < oh; ++y)
                    std::memcpy(out.ptr() + ((n * x.c() + c) * oh + y) * ow,
                                x.ptr() + ((n * x.c() + c) * x.h() + y0 + y) * x.w() + x0,
                                ow * sizeof(float));
        return out;
    }

    Tensor tiled_encode(const Tensor& x) {
        auto t = x;
        int orig_h = x.h(), orig_w = x.w();
        bool qwen = opt_.model == "vosr2";
        auto latent_dims = [qwen](int hh, int ww) {
            if (qwen) return std::pair<int,int>{hh / AE_FACTOR, ww / AE_FACTOR};
            return std::pair<int,int>{(hh + AE_FACTOR - 1) / AE_FACTOR,
                                      (ww + AE_FACTOR - 1) / AE_FACTOR};
        };
        auto out_hw = latent_dims(orig_h, orig_w);
        int out_h = out_hw.first, out_w = out_hw.second;
        // 单块路径直接喂原图：sd2/qwen 的 VAE 各自处理 padding 与 floor 规则。
        if (opt_.tile_size <= 0) return crop(enc_.forward({t}), out_h, out_w);
        int ph = (8 - orig_h % 8) % 8, pw = (8 - orig_w % 8) % 8;
        if (ph || pw) {
            Tensor padded = zeros({1, 3, orig_h + ph, orig_w + pw});
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < orig_h + ph; ++y)
                    std::memcpy(padded.ptr() + (c * (orig_h + ph) + y) * (orig_w + pw),
                                x.ptr() + (c * orig_h + std::min(y, orig_h - 1)) * orig_w,
                                orig_w * sizeof(float));
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < orig_h + ph; ++y) {
                    float v = padded.data[(c * (orig_h + ph) + y) * (orig_w + pw) + orig_w - 1];
                    for (int xx = orig_w; xx < orig_w + pw; ++xx)
                        padded.data[(c * (orig_h + ph) + y) * (orig_w + pw) + xx] = v;
                }
            t = padded;
        }
        int lh = t.h() / AE_FACTOR, lw = t.w() / AE_FACTOR;
        int ts = std::max(opt_.tile_size / AE_FACTOR, 1);
        int to = std::min(std::max(opt_.tile_overlap / AE_FACTOR, ts / 8), ts - 1);
        ts = std::min(ts, std::min(lh, lw));
        auto hs = tile_grid(lh, ts, to), ws = tile_grid(lw, ts, to);
        if (hs.size() == 1 && ws.size() == 1)
            return crop(enc_.forward({t}), out_h, out_w);
        log("VAE encode 分块: latent " + std::to_string(lh) + "x" + std::to_string(lw) +
            ", tile=" + std::to_string(ts) + ", grid=" + std::to_string(hs.size()) + "x" + std::to_string(ws.size()));
        int lc = opt_.latent_channels;
        auto acc = zeros({1, lc, lh, lw}), wacc = zeros({1, lc, lh, lw});
        auto g = gaussian_weights(ts);
        for (int hi : hs) {
            for (int wi : ws) {
                int he = std::min(hi + ts, lh), we = std::min(wi + ts, lw);
                auto crop_x = region(t, hi * AE_FACTOR, he * AE_FACTOR, wi * AE_FACTOR, we * AE_FACTOR);
                auto zt = enc_.forward({crop_x});
                for (int c = 0; c < lc; ++c)
                    for (int y = 0; y < he - hi; ++y)
                        for (int x = 0; x < we - wi; ++x) {
                            float gw = g[y * ts + x];
                            size_t idx = (static_cast<size_t>(c) * lh + hi + y) * lw + wi + x;
                            acc.data[idx] += zt.data[(c * (he - hi) + y) * (we - wi) + x] * gw;
                            wacc.data[idx] += gw;
                        }
            }
        }
        for (size_t i = 0; i < acc.size(); ++i) acc.data[i] /= std::max(wacc.data[i], 1e-12f);
        return crop(acc, out_h, out_w);
    }

    Tensor tiled_decode(const Tensor& z) {
        if (opt_.tile_size <= 0) return dec_.forward({z});
        int h = z.h(), w = z.w();
        int ts = std::max(opt_.tile_size / AE_FACTOR, 1);
        int to = std::min(std::max(opt_.tile_overlap / AE_FACTOR, ts / 8), ts - 1);
        ts = std::min(ts, std::min(h, w));
        auto hs = tile_grid(h, ts, to), ws = tile_grid(w, ts, to);
        if (hs.size() == 1 && ws.size() == 1) return dec_.forward({z});
        int oh = h * AE_FACTOR, ow = w * AE_FACTOR;
        log("VAE decode 分块: latent " + std::to_string(h) + "x" + std::to_string(w) +
            ", tile=" + std::to_string(ts) + ", grid=" + std::to_string(hs.size()) + "x" + std::to_string(ws.size()));
        auto acc = zeros({1, 3, oh, ow}), wacc = zeros({1, 3, oh, ow});
        auto gp = gaussian_weights(ts * AE_FACTOR);
        int gsize = ts * AE_FACTOR;
        for (int hi : hs) {
            for (int wi : ws) {
                int he = std::min(hi + ts, h), we = std::min(wi + ts, w);
                auto zt = dec_.forward({region(z, hi, he, wi, we)});
                int ph = (he - hi) * AE_FACTOR, pw = (we - wi) * AE_FACTOR;
                int py0 = hi * AE_FACTOR, px0 = wi * AE_FACTOR;
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < ph; ++y)
                        for (int x = 0; x < pw; ++x) {
                            float gw = gp[y * gsize + x];
                            size_t idx = (static_cast<size_t>(c) * oh + py0 + y) * ow + px0 + x;
                            acc.data[idx] += zt.data[(c * ph + y) * pw + x] * gw;
                            wacc.data[idx] += gw;
                        }
            }
        }
        for (size_t i = 0; i < acc.size(); ++i) acc.data[i] /= std::max(wacc.data[i], 1e-12f);
        return acc;
    }

    const Options& opt_;
    MNNModel enc_, dino_, dit_, dec_;
};

} // namespace

bool load_image(const std::string& path, Image& img) {
    int n = 0;
    uint8_t* p = stbi_load(path.c_str(), &img.w, &img.h, &n, 3);
    if (!p) return false;
    img.c = 3;
    img.data.assign(p, p + static_cast<size_t>(img.w) * img.h * 3);
    stbi_image_free(p);
    return true;
}

bool save_image(const std::string& path, const Image& img) {
    return stbi_write_png(path.c_str(), img.w, img.h, 3, img.data.data(), img.w * 3) != 0;
}

Image infer(const Options& opt, const Image& input, const std::string& output_path) {
    MNNForwardType forward = MNN_FORWARD_CPU;
    if (opt.backend == "opencl") {
        forward = MNN_FORWARD_OPENCL;
    } else if (opt.backend != "cpu") {
        throw std::runtime_error("未知后端: " + opt.backend + "（可选 cpu|opencl）");
    }
    MNN::ScheduleConfig cfg;
    cfg.type = forward;
    if (forward == MNN_FORWARD_OPENCL) {
        // numThread 与 mode 是 union：GPU 后端把它当作 gpuMode 位掩码。
        // 默认强制 BUFFER 模式，绕开 cl::Image2D 的宽高上限（大张量如 4x37632 会
        // 触发 CL_INVALID_IMAGE_SIZE，且 STATIC 权重分配早于尺寸检查回退）。
        if (opt.opencl_memory == "buffer") {
            cfg.numThread = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE;
        } else if (opt.opencl_memory == "image") {
            cfg.numThread = MNN_GPU_MEMORY_IMAGE | MNN_GPU_TUNING_WIDE;
        } else {
            throw std::runtime_error("未知 --opencl-memory: " + opt.opencl_memory + "（可选 buffer|image）");
        }
    } else {
        cfg.numThread = opt.threads;
    }
    cfg.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig bn;
    bn.precision = MNN::BackendConfig::Precision_Normal;
    bn.memory = MNN::BackendConfig::Memory_Normal;
    bn.power = MNN::BackendConfig::Power_Normal;
    cfg.backendConfig = &bn;
    auto rt = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
        MNN::Express::Executor::RuntimeManager::createRuntimeManager(cfg));
    if (!rt) throw std::runtime_error("创建 MNN RuntimeManager 失败");
    log("MNN 后端: " + opt.backend +
        (forward == MNN_FORWARD_OPENCL ? " (" + opt.opencl_memory + ")" : "") +
        ", 线程=" + std::to_string(opt.threads));

    Engine engine(opt, rt);
    auto start = std::chrono::steady_clock::now();
    auto out = engine.run(input);
    auto sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    log("推理完成，用时 " + std::to_string(sec) + "s");
    if (!output_path.empty() && !save_image(output_path, out))
        throw std::runtime_error("保存输出图片失败: " + output_path);
    if (!output_path.empty()) log("保存 " + output_path);
    return out;
}

} // namespace vosr
