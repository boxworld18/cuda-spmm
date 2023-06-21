#include "gtest/gtest.h"
#include "util.h"
#include "data.h"

int main(int argc, char** argv)
{
    argParse(argc, argv);
    load_graph(inputgraph, kNumV, kNumE, gptr_cpu, gidx_cpu);
    checkCudaErrors(cudaMalloc2((void**)&gptr, (kNumV + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&gidx, kNumE * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gptr, gptr_cpu, sizeof(int) * (kNumV + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gidx, gidx_cpu, sizeof(int) * kNumE, cudaMemcpyHostToDevice));
    registerPtr(gptr);
    registerPtr(gidx);
    dbg(kLen);

    curandCreateGenerator(&kCuRand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(kCuRand, 123ULL);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}