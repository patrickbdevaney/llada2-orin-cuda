// C API of the engine (libllada.so), used by server/ (HTTP, tokenizer, chat template) through ctypes.
//   nvcc -O3 -arch=sm_87 -std=c++17 -Xcompiler -fPIC -shared libllada.cu -o libllada.so
#include <chrono>
#include <mutex>
#include "decode.cu"

struct Handle {
    eng::Engine e;
    dec::Decoder* d = nullptr;
    std::mutex mu;
};

extern "C" {

void* llada_init(const char* dir, int max_ctx) {
    auto* h = new Handle();
    h->e.load(dir, max_ctx);
    h->d = new dec::Decoder(h->e);
    return h;
}

// Generate after `prompt`; out receives up to out_cap ids (EOS included when produced). on_block (may be null) is
// called with every decoded block's new tokens. Returns the number of ids written.
// threshold / editing_threshold / max_post_steps / steps: JointThresholdInDel knobs (model card defaults 0.5 / 0.0 /
// 16 / 32). steps sets the per-step unmasking floor (block_length / steps tokens): fewer steps, faster, lower quality.
int llada_generate(void* hp, const int* prompt, int n, int max_new, float threshold, float editing_threshold,
                   int max_post_steps, int steps, int* out, int out_cap, void (*on_block)(const int*, int, void*),
                   void* user) {
    auto* h = static_cast<Handle*>(hp);
    std::lock_guard<std::mutex> lk(h->mu);
    h->d->P.threshold = threshold;
    h->d->P.editing_threshold = editing_threshold;
    h->d->P.max_post_steps = max_post_steps;
    h->d->P.steps = std::max(1, std::min(32, steps));
    const int max_ctx = (h->e.max_blocks - 1) * eng::T;
    if (n + max_new > max_ctx) max_new = std::max(0, max_ctx - n);
    std::vector<int> p(prompt, prompt + n);
    auto ids = h->d->generate(p, max_new, true, [&](const int* t, int k) { if (on_block) on_block(t, k, user); });
    const int m = std::min<int>(int(ids.size()), out_cap);
    std::copy(ids.begin(), ids.begin() + m, out);
    return m;
}

// stats: [steps, blocks, commits, reused_blocks_last, weight_GB, kv_GB, max_ctx]
void llada_stats(void* hp, double* s) {
    auto* h = static_cast<Handle*>(hp);
    s[0] = double(h->d->S.steps); s[1] = double(h->d->S.blocks); s[2] = double(h->d->S.commits);
    s[3] = h->d->reused_blocks; s[4] = h->e.weight_bytes / 1e9; s[5] = h->e.kv_bytes / 1e9;
    s[6] = (h->e.max_blocks - 1) * eng::T;
}

}
