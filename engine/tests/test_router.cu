// router_mma logits vs an fp64 host reference on a real layer's router input.   ./test_router <engine dir> [layer]
#include "decode.cu"
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    const int li = argc > 2 ? atoi(argv[2]) : 10;
    eng::Engine e;
    e.load(dir, 4096);
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    e.begin_block(ids.data(), 0);
    for (int l = 0; l <= li; ++l) e.layer_step(l, 0);
    CK(cudaStreamSynchronize(e.st));
    std::vector<float> hn(32 * 2048), part(ops::ROUTER_KC * 32 * 256);
    std::vector<__nv_bfloat16> W(256 * 2048);
    CK(cudaMemcpy(hn.data(), e.hn, hn.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(W.data(), e.layers[li].router, W.size() * 2, cudaMemcpyDeviceToHost));
    ops::router_mma<<<dim3(ops::ROUTER_KC, 1), 256, ops::ROUTER_SMEM, e.st>>>(e.hn, e.layers[li].router, e.rlogits);
    CK(cudaMemcpy(part.data(), e.rlogits, part.size() * 4, cudaMemcpyDeviceToHost));
    double maxd = 0, maxf = 0, rms = 0;
    for (int t = 0; t < 32; ++t)
        for (int x = 0; x < 256; ++x) {
            double r = 0; float f = 0.f;
            for (int k = 0; k < 2048; ++k) { r += double(hn[t * 2048 + k]) * double(__bfloat162float(W[x * 2048 + k])); f += hn[t * 2048 + k] * __bfloat162float(W[x * 2048 + k]); }
            float g = 0.f; for (int kc = 0; kc < ops::ROUTER_KC; ++kc) g += part[(kc * 32 + t) * 256 + x];
            maxd = std::max(maxd, std::fabs(g - r)); maxf = std::max(maxf, std::fabs(double(f) - r)); rms += r * r;
        }
    printf("layer %d: logit rms %.3f | max |mma - fp64| %.2e | max |sequential fp32 - fp64| %.2e\n", li, std::sqrt(rms / 8192), maxd, maxf);
    return 0;
}
