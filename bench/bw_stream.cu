#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
__global__ void gstride(const int4* __restrict__ a, size_t n, int* out){
  int4 acc={0,0,0,0};
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){int4 v=__ldg(a+i);acc.x^=v.x;acc.y^=v.y;acc.z^=v.z;acc.w^=v.w;}
  if((acc.x^acc.y^acc.z^acc.w)==0x12345) *out=1;
}
// each block streams a contiguous chunk (like a GEMV tile over weight rows), 4 loads in flight
__global__ void chunk(const int4* __restrict__ a, size_t n, int* out){
  size_t per=(n+gridDim.x-1)/gridDim.x, s=blockIdx.x*per, e=min(n,s+per);
  int4 acc={0,0,0,0};
  for(size_t i=s+threadIdx.x;i<e;i+=4*blockDim.x){
    int4 v[4];
    #pragma unroll
    for(int u=0;u<4;u++){size_t j=i+u*blockDim.x; v[u]= j<e? __ldg(a+j): make_int4(0,0,0,0);}
    #pragma unroll
    for(int u=0;u<4;u++){acc.x^=v[u].x;acc.y^=v[u].y;acc.z^=v[u].z;acc.w^=v[u].w;}
  }
  if((acc.x^acc.y^acc.z^acc.w)==0x12345) *out=1;
}
int main(){
  int* o; cudaMalloc(&o,4);
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  for(size_t mb: {64,256,512,1024,2048,3072}){
    size_t bytes=mb<<20; int4* a; if(cudaMalloc(&a,bytes)!=cudaSuccess){printf("%zuMB alloc fail\n",mb);continue;} cudaMemset(a,1,bytes);
    for(int k=0;k<2;k++) for(int g: {64,256,1024}){
      auto run=[&](){ if(k==0) gstride<<<g,512>>>(a,bytes/16,o); else chunk<<<g,256>>>(a,bytes/16,o);};
      for(int w=0;w<3;w++) run();
      int it= mb<=512? 40:10; cudaEventRecord(s); for(int i=0;i<it;i++) run(); cudaEventRecord(e); cudaEventSynchronize(e);
      float ms; cudaEventElapsedTime(&ms,s,e); printf("%5zuMB %-7s grid=%4d: %6.1f GB/s\n",mb,k?"chunk":"gstride",g,bytes*(double)it/ms/1e6);}
    cudaFree(a);}
  // weights split in many 128MB buffers read back-to-back (realistic arena of layer tensors)
  std::vector<int4*> bufs; size_t B=128<<20; for(int i=0;i<24;i++){int4* p; if(cudaMalloc(&p,B)!=cudaSuccess)break; cudaMemset(p,1,B); bufs.push_back(p);}
  for(int w=0;w<2;w++) for(auto p:bufs) chunk<<<256,256>>>(p,B/16,o);
  cudaEventRecord(s); for(int r=0;r<5;r++) for(auto p:bufs) chunk<<<256,256>>>(p,B/16,o); cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms,s,e); printf("%zu x 128MB buffers sequential (%.1f GB): %.1f GB/s (%.1f us/launch)\n",bufs.size(),bufs.size()*B/1e9,5.0*bufs.size()*B/ms/1e6, ms*1000/(5*bufs.size()));
}
