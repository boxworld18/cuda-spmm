#ifndef UTIL_H
#define UTIL_H

#include <iostream>
//#include <cuda_profiler_api.h>
#include <curand.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>
#include <assert.h>
#include <sstream>
#include <assert.h>
#include <pthread.h>
#include <omp.h>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <random>

#include "dbg.h"
#include "args.hxx"

using namespace std;

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// extern ncclComm_t* comms;
extern int kNumV, kNumE, kLen;
extern vector<void *> registered_ptr;
extern size_t total_size;
extern string basedir;
extern string inputgraph;
extern string edgefile;
extern string ptrfile;
extern string dataset;

extern curandGenerator_t kCuRand;


// ************************************************************
// variables for single train
extern int *gptr, *gidx;

extern int *gptr_cpu, *gidx_cpu;


inline bool fexist(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
void argParse(int argc, char **argv, int *p_limit = NULL, int *p_limit2 = NULL);

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

#define FatalError(s)                                     \
    do                                                    \
    {                                                     \
        std::stringstream _where, _message;               \
        _where << __FILE__ << ':' << __LINE__;            \
        _message << std::string(s) + "\n"                 \
                 << __FILE__ << ':' << __LINE__;          \
        std::cerr << _message.str() << "\nAborting...\n"; \
        cudaDeviceReset();                                \
        exit(1);                                          \
    } while (0)

#define checkCudaErrors(status)                   \
    do                                            \
    {                                             \
        std::stringstream _error;                 \
        if (status != 0)                          \
        {                                         \
            _error << "Cuda failure: " << status; \
            FatalError(_error.str());             \
        }                                         \
    } while (0)

inline cudaError_t cudaMalloc2(void **a, size_t s)
{
    if (s == 0)
        return (cudaError_t)(0);
    total_size += s;
    return cudaMalloc(a, ((s + 511) / 512) * 512);
    //return cudaMallocManaged(a, ((s + 511) / 512) * 512);
}

template <class T>
inline void registerPtr(T ptr)
{
    registered_ptr.push_back((void *)(ptr));
}

template <class T>
void safeFree(T *&a)
{
    for (auto item : registered_ptr)
    {
        if (((void *)(a)) == item)
            return;
    }
    if (a != NULL)
    {
        cudaFree(a);
        cudaGetLastError();
        a = NULL;
    }
    else
    {
    }
}

struct CSR
{
    CSR(int out_num_v, int out_num_e, int *outptr, int *outidx, float *outval) : num_v(out_num_v), num_e(out_num_e), ptr(outptr), idx(outidx), val(outval) {}

    int num_v = 0;
    int num_e = 0;
    int *ptr = NULL;
    int *idx = NULL;
    float *val = NULL;
};
 
inline double getCUDATime(const std::function<void()> &f)
{
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t0);
    f();
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t1);
    return getDuration(t0, t1);
}

inline double getAverageTimeWithWarmUp(const std::function<void()> &f)
{
    int n_warmup = 5;
    int n_run = 20;
    for (int i = 0; i < n_warmup; ++i)
    f();
    double total_time = 0;
    int i = 0;
    for (i = 0; i < n_run; ++i) {
        total_time += getCUDATime(f);
        if (total_time > 10) 
        {
            ++i;
            break;
        }
    }
    return total_time / i;
}

#endif
