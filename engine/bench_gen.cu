// Generation benchmark: ./bench_gen <engine dir> <prompt ids file (int32)> [max_new] [max_ctx]
// Prints generated ids (one line) and timing: prefill, decode tok/s, steps per block; then repeats the same
// request to show the prefix-cache path.
#include <chrono>
#include "decode.cu"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: bench_gen <engine dir> <prompt.bin> [max_new] [max_ctx]\n"); return 1; }
    const int max_new = argc > 3 ? atoi(argv[3]) : 256, max_ctx = argc > 4 ? atoi(argv[4]) : 8192;
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "no %s\n", argv[2]); return 1; }
    std::vector<int> prompt;
    int x;
    while (fread(&x, 4, 1, f) == 1) prompt.push_back(x);
    fclose(f);
    auto t0 = std::chrono::steady_clock::now();
    eng::Engine e;
    e.load(argv[1], max_ctx);
    auto t1 = std::chrono::steady_clock::now();
    printf("load %.1fs: %.2f GB weights, %.2f GB KV (max ctx %d)\n", std::chrono::duration<double>(t1 - t0).count(),
           e.weight_bytes / 1e9, e.kv_bytes / 1e9, max_ctx);
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    printf("device memory free %.2f / %.2f GB\n", fr / 1e9, tot / 1e9);
    dec::Decoder d(e);
    if (getenv("PREFILL_BENCH")) {                // prefill only: all full prompt blocks, per-section profile
        const int full = int(prompt.size()) / 32, pb = getenv("PBT") ? atoi(getenv("PBT")) : eng::PB;
        e.profiling = true;
        for (int rep = 0; rep < 2; ++rep) {
            e.prof_acc.clear();
            auto a = std::chrono::steady_clock::now();
            for (int b = 0; b < full; b += pb) { e.prefill(prompt.data() + b * 32, std::min(pb, full - b), b); e.prof_collect(); }
            CK(cudaStreamSynchronize(e.st));
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            printf("prefill %d tokens in %.3fs: %.0f tok/s (batch %d blocks)\n", full * 32, dt, full * 32 / dt, pb);
        }
        static const char* nm[] = {"(gap)", "norm+qkv", "qk-norm+rope+commit", "attention", "quant+dense", "layer0 mlp",
                                   "route", "moe gate_up", "fwht512", "moe down", "combine", "-", "-", "-", "-",
                                   "embed", "post norm+rot+q8", "router logits"};
        double tot = 0;
        for (double v : e.prof_acc) tot += v;
        for (size_t k = 0; k < e.prof_acc.size(); ++k)
            if (e.prof_acc[k] > 0) printf("  %-20s %8.1f ms %5.1f%%\n", nm[k], e.prof_acc[k], 100 * e.prof_acc[k] / tot);
        return 0;
    }
    e.profiling = getenv("PROF") != nullptr;
    if (getenv("LLADA_THRESHOLD")) d.P.threshold = float(atof(getenv("LLADA_THRESHOLD")));
    if (getenv("LLADA_STEPS")) d.P.steps = atoi(getenv("LLADA_STEPS"));
    for (int rep = 0; rep < 2; ++rep) {
        const auto s0 = d.S;
        auto a = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point first;
        bool got_first = false;
        auto out = d.generate(prompt, max_new, true, [&](const int*, int) {
            if (!got_first) { first = std::chrono::steady_clock::now(); got_first = true; }
        });
        auto b = std::chrono::steady_clock::now();
        const double total = std::chrono::duration<double>(b - a).count();
        const double ttfb = got_first ? std::chrono::duration<double>(first - a).count() : total;
        const long steps = d.S.steps - s0.steps, blocks = d.S.blocks - s0.blocks, commits = d.S.commits - s0.commits;
        if (rep == 0) {
            for (int v : out) printf("%d ", v);
            printf("\n");
        }
        printf("[run %d] prompt %zu tok (%d blocks reused), %zu generated in %.2fs: first block %.2fs, "
               "%.1f tok/s overall, %ld denoising steps (%.1f / block, %.2f tok / step), %ld commit passes (%ld kv-only, %ld prefill passes)\n",
               rep, prompt.size(), d.reused_blocks, out.size(), total, ttfb, out.size() / total, steps,
               blocks ? double(steps) / blocks : 0.0, steps ? double(out.size()) / steps : 0.0, commits,
               d.S.kv_only_commits - s0.kv_only_commits, d.S.prefill_passes - s0.prefill_passes);
    }
    if (e.profiling) {
        static const char* nm[] = {"(gap)", "norm+qkv", "qk-norm+rope", "attention", "quant+dense", "layer0 mlp",
                                   "route_schedule", "moe gate_up", "fwht512", "moe down", "combine", "kv commit",
                                   "final norm+lm_head", "logits_top", "-", "embed", "post norm+rot+q8", "router logits"};
        double tot = 0;
        for (double v : e.prof_acc) tot += v;
        const long passes = d.S.steps + d.S.commits - d.S.kv_only_commits;
        printf("profile over %ld forward passes (%.2f ms/pass):\n", passes, tot / passes);
        for (size_t k = 0; k < e.prof_acc.size(); ++k)
            if (e.prof_acc[k] > 0) printf("  %-20s %8.2f ms/pass %5.1f%%\n", nm[k], e.prof_acc[k] / passes, 100 * e.prof_acc[k] / tot);
    }
    return 0;
}
