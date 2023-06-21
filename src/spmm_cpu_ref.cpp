#include "spmm_cpu_ref.h"

void run_spmm_cpu(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len)
{
    for (int i = 0; i < num_v; ++i)
    {
        for (int j = ptr[i]; j < ptr[i + 1]; ++j)
        {
            for (int k = 0; k < feat_len; ++k)
            {
                vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
            }
        }
    }
}

void SpMMCPURef::preprocess(float *vin, float *vout)
{
}

void SpMMCPURef::run(float *vin, float *vout)
{
    run_spmm_cpu(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
