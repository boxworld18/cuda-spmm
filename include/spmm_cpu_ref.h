#pragma once
#ifndef SpMM_CPU_REF_H
#define SpMM_CPU_REF_H
#include "spmm_base.h"

class SpMMCPURef : public SpMM
{
public:
    SpMMCPURef(int *out_ptr, int *out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(out_ptr, out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMCPURef(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}

    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);
};
#endif