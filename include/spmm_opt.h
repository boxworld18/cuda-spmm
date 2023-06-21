#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
        if (row_index) checkCudaErrors(cudaFree(row_index));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    int num_target;
    int *target, *ptr_scheduled;
    int *row_index = NULL;
    int *pd_ptr, *pd_idx;
    float *pd_val;

    int *new_ptr, *new_idx;
    float *new_val;

    int mode = 0;
    
    int *lar_ptr, *large_row_index, *small_row_index;
    int lr_cnt, sr_cnt;
    
    dim3 large_grid;
    dim3 large_block;
};
#endif