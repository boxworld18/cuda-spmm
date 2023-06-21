#include "spmm_cpu_opt.h"

void spmm_cpu_256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len) {
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < num_v; i++) {
        int out_base = i * feat_len;

        for (int j = ptr[i]; j < ptr[i + 1]; j++) {
            int in_base = idx[j] * feat_len;
            
            float myval = val[j];
            float *vout_ptr = vout + out_base;
            float *vin_ptr = vin + in_base;

            for (int k = 0; k < feat_len; k += 32) {
                vout_ptr[k] += vin_ptr[k] * myval;
                vout_ptr[k + 1] += vin_ptr[k + 1] * myval;
                vout_ptr[k + 2] += vin_ptr[k + 2] * myval;
                vout_ptr[k + 3] += vin_ptr[k + 3] * myval;
                vout_ptr[k + 4] += vin_ptr[k + 4] * myval;
                vout_ptr[k + 5] += vin_ptr[k + 5] * myval;
                vout_ptr[k + 6] += vin_ptr[k + 6] * myval;
                vout_ptr[k + 7] += vin_ptr[k + 7] * myval;
                vout_ptr[k + 8] += vin_ptr[k + 8] * myval;
                vout_ptr[k + 9] += vin_ptr[k + 9] * myval;
                vout_ptr[k + 10] += vin_ptr[k + 10] * myval;
                vout_ptr[k + 11] += vin_ptr[k + 11] * myval;
                vout_ptr[k + 12] += vin_ptr[k + 12] * myval;
                vout_ptr[k + 13] += vin_ptr[k + 13] * myval;
                vout_ptr[k + 14] += vin_ptr[k + 14] * myval;
                vout_ptr[k + 15] += vin_ptr[k + 15] * myval;
                vout_ptr[k + 16] += vin_ptr[k + 16] * myval;
                vout_ptr[k + 17] += vin_ptr[k + 17] * myval;
                vout_ptr[k + 18] += vin_ptr[k + 18] * myval;
                vout_ptr[k + 19] += vin_ptr[k + 19] * myval;
                vout_ptr[k + 20] += vin_ptr[k + 20] * myval;
                vout_ptr[k + 21] += vin_ptr[k + 21] * myval;
                vout_ptr[k + 22] += vin_ptr[k + 22] * myval;
                vout_ptr[k + 23] += vin_ptr[k + 23] * myval;
                vout_ptr[k + 24] += vin_ptr[k + 24] * myval;
                vout_ptr[k + 25] += vin_ptr[k + 25] * myval;
                vout_ptr[k + 26] += vin_ptr[k + 26] * myval;
                vout_ptr[k + 27] += vin_ptr[k + 27] * myval;
                vout_ptr[k + 28] += vin_ptr[k + 28] * myval;
                vout_ptr[k + 29] += vin_ptr[k + 29] * myval;
                vout_ptr[k + 30] += vin_ptr[k + 30] * myval;
                vout_ptr[k + 31] += vin_ptr[k + 31] * myval;
            }
        }
    }
}

void spmm_cpu_32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len) {
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < num_v; i++) {
        int out_base = i * feat_len;

        for (int j = ptr[i]; j < ptr[i + 1]; j++) {
            int in_base = idx[j] * feat_len;
            
            float myval = val[j];
            float *vout_ptr = vout + out_base;
            float *vin_ptr = vin + in_base;

            vout_ptr[0] += vin_ptr[0] * myval;
            vout_ptr[1] += vin_ptr[1] * myval;
            vout_ptr[2] += vin_ptr[2] * myval;
            vout_ptr[3] += vin_ptr[3] * myval;
            vout_ptr[4] += vin_ptr[4] * myval;
            vout_ptr[5] += vin_ptr[5] * myval;
            vout_ptr[6] += vin_ptr[6] * myval;
            vout_ptr[7] += vin_ptr[7] * myval;
            vout_ptr[8] += vin_ptr[8] * myval;
            vout_ptr[9] += vin_ptr[9] * myval;
            vout_ptr[10] += vin_ptr[10] * myval;
            vout_ptr[11] += vin_ptr[11] * myval;
            vout_ptr[12] += vin_ptr[12] * myval;
            vout_ptr[13] += vin_ptr[13] * myval;
            vout_ptr[14] += vin_ptr[14] * myval;
            vout_ptr[15] += vin_ptr[15] * myval;
            vout_ptr[16] += vin_ptr[16] * myval;
            vout_ptr[17] += vin_ptr[17] * myval;
            vout_ptr[18] += vin_ptr[18] * myval;
            vout_ptr[19] += vin_ptr[19] * myval;
            vout_ptr[20] += vin_ptr[20] * myval;
            vout_ptr[21] += vin_ptr[21] * myval;
            vout_ptr[22] += vin_ptr[22] * myval;
            vout_ptr[23] += vin_ptr[23] * myval;
            vout_ptr[24] += vin_ptr[24] * myval;
            vout_ptr[25] += vin_ptr[25] * myval;
            vout_ptr[26] += vin_ptr[26] * myval;
            vout_ptr[27] += vin_ptr[27] * myval;
            vout_ptr[28] += vin_ptr[28] * myval;
            vout_ptr[29] += vin_ptr[29] * myval;
            vout_ptr[30] += vin_ptr[30] * myval;
            vout_ptr[31] += vin_ptr[31] * myval;   
        }
    }
}

void SpMMCPUOpt::preprocess(float *vin, float *vout) {
    if (feat_in == 32) mode = 1;
    if (feat_in == 256) mode = 2;
}

void SpMMCPUOpt::run(float *vin, float *vout) {
    switch (mode) {
        case 1:
            spmm_cpu_32(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
            break;
        case 2:
        default:
            spmm_cpu_256(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}
