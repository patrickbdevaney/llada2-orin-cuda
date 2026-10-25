// Engine layers vs the BF16 HF reference (gen_layer_case.py): the context's reference K/V are committed into the
// 4c/4 cache, then the engine runs its layer steps on the block and the block's hidden state is compared after
// every layer (relative L2 error and cosine), plus top-8 routing overlap for MoE layers.
//   ./test_layers <model dir> <case dir> [self]   (self: the engine commits the context itself)
#include "engine.cu"

template <class X> std::vector<X> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / sizeof(X); fseek(f, 0, SEEK_SET);
    std::vector<X> v(n);
    if (fread(v.data(), sizeof(X), n, f) != n) exit(1);
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const std::string mdir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    const std::string cdir = argc > 2 ? argv[2] : "layers";
    std::ifstream cj(cdir + "/case.json");
    std::string js((std::istreambuf_iterator<char>(cj)), std::istreambuf_iterator<char>());
    auto num = [&](const std::string& k) { return std::stoi(js.substr(js.find("\"" + k + "\"") + k.size() + 3)); };
    const int ctx = num("ctx"), nl = num("layers"), cb = ctx / 32;
    eng::Engine e;
    e.load(mdir, 4096, nl);
    printf("loaded %d layers: %.2f GB weights, %.2f GB KV\n", nl, e.weight_bytes / 1e9, e.kv_bytes / 1e9);
    auto ids = rd<int>(cdir + "/block_ids.bin");
    const std::string mode = argc > 3 ? argv[3] : "";
    const bool self_commit = mode == "self" || mode == "batch";
    if (self_commit) {                      // the engine's own commit passes over the context (serving path)
        auto ctx_ids = rd<int>(cdir + "/ctx_ids.bin");
        if (mode == "batch")
            for (int b = 0, pb = getenv("PBT") ? atoi(getenv("PBT")) : eng::PB; b < cb; b += pb) e.prefill(ctx_ids.data() + b * 32, std::min(pb, cb - b), b);
        else
            for (int b = 0; b < cb; ++b) e.forward_block(ctx_ids.data() + b * 32, b * 32, b, true, false);
        printf("context committed by the engine itself (%d blocks, %s)\n", cb, mode.c_str());
    }
    for (int li = 0; li < nl && !self_commit; ++li) {
        auto k = rd<uint16_t>(cdir + "/k_L" + std::to_string(li) + ".bin");
        auto v = rd<uint16_t>(cdir + "/v_L" + std::to_string(li) + ".bin");
        for (int b = 0; b < cb; ++b) {
            CK(cudaMemcpy(e.layers[li].kf, k.data() + size_t(b) * 32 * 4 * 128, 32 * 4 * 128 * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(e.layers[li].vf, v.data() + size_t(b) * 32 * 4 * 128, 32 * 4 * 128 * 2, cudaMemcpyHostToDevice));
            attn::kv_commit<<<4, 128>>>(e.layers[li].kf, e.layers[li].vf, e.layers[li].kv, e.max_blocks, b);
        }
    }
    CK(cudaDeviceSynchronize());
    e.begin_block(ids.data(), ctx);
    std::vector<float> h(32 * 2048);
    int fails = 0;
    for (int li = 0; li < nl; ++li) {
        e.layer_step(li, cb);
        CK(cudaStreamSynchronize(e.st));
        CK(cudaGetLastError());
        CK(cudaMemcpy(h.data(), e.h, h.size() * 4, cudaMemcpyDeviceToHost));
        auto ref = rd<float>(cdir + "/h_L" + std::to_string(li) + ".bin");
        double n = 0, d = 0, dot = 0, a2 = 0;
        for (size_t i = 0; i < h.size(); ++i) {
            n += (h[i] - ref[i]) * double(h[i] - ref[i]); d += double(ref[i]) * ref[i];
            dot += double(h[i]) * ref[i]; a2 += double(h[i]) * h[i];
        }
        const double rel = std::sqrt(n / d), cosv = dot / std::sqrt(a2 * d);
        if (getenv("PER_TOKEN") && li == atoi(getenv("PER_TOKEN"))) {
            for (int t = 0; t < 32; ++t) {
                double nn = 0, dd = 0;
                for (int c = 0; c < 2048; ++c) { const double df = h[t * 2048 + c] - ref[t * 2048 + c]; nn += df * df; dd += double(ref[t * 2048 + c]) * ref[t * 2048 + c]; }
                printf("   token %2d id %6d: rel %.4f |ref| %.2f\n", t, ids[t], std::sqrt(nn / dd), std::sqrt(dd / 2048));
            }
        }
        printf("layer %d: block hidden rel L2 err %.4f, cosine %.5f", li, rel, cosv);
        auto tk = rd<int>(cdir + "/topk_L" + std::to_string(li) + ".bin");
        if (!tk.empty() && e.layers[li].moe) {
            std::vector<int> mine(32 * 8);
            CK(cudaMemcpy(mine.data(), e.sched.topk, mine.size() * 4, cudaMemcpyDeviceToHost));
            int agree = 0;
            for (int t = 0; t < 32; ++t)
                for (int a = 0; a < 8; ++a)
                    for (int b = 0; b < 8; ++b) agree += mine[t * 8 + a] == tk[t * 8 + b];
            printf(", top-8 routing overlap %.3f", agree / 256.0);
        }
        printf("\n");
        fails += rel > 0.1;
    }
    return fails ? 2 : 0;
}
