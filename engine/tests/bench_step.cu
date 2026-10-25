// Steady-state cost of one denoising step: the decoder's captured step graph replayed at a fixed context length.
//   ./bench_step <engine dir> [ctx_blocks ...]      (default 4 32 100)
#include "decode.cu"
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    std::vector<int> ctxs;
    for (int i = 2; i < argc; ++i) ctxs.push_back(atoi(argv[i]));
    if (ctxs.empty()) ctxs = {4, 32, 100};
    const int maxcb = *std::max_element(ctxs.begin(), ctxs.end());
    eng::Engine e; e.load(dir, (maxcb + eng::PB + 1) * 32);
    dec::Decoder d(e);
    std::vector<int> ids(32); for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    for (int b = 0; b < maxcb; b += eng::PB)
        e.prefill(std::vector<int>(32 * eng::PB, 1234).data(), eng::PB, b);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int cb : ctxs) {
        d.ctx_blocks = cb;
        d.step_logits(ids);                                     // capture / patch
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            CK(cudaEventRecord(a, e.st));
            for (int i = 0; i < 40; ++i) CK(cudaGraphLaunch(d.G.exec, e.st));
            CK(cudaEventRecord(b, e.st)); CK(cudaEventSynchronize(b));
            float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, double(ms) / 40);
        }
        size_t nn = 0; CK(cudaGraphGetNodes(d.G.graph, nullptr, &nn));
        std::vector<cudaGraphNode_t> nodes(nn); CK(cudaGraphGetNodes(d.G.graph, nodes.data(), &nn));
        int nk = 0; for (auto nd : nodes) { cudaGraphNodeType ty; CK(cudaGraphNodeGetType(nd, &ty)); nk += ty == cudaGraphNodeTypeKernel; }
        printf("ctx %4d tokens: %.3f ms/step  (graph: %zu nodes, %d kernels)\n", cb * 32, best, nn, nk);
    }
    return 0;
}
