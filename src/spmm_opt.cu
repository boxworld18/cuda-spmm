#include <algorithm>
#include <iostream>
#include "spmm_opt.h"

typedef pair<int, int> pii;

constexpr int k_block_x = 4;
constexpr int k_block_y = 32;
constexpr int sm256_y = 128;
constexpr int k_block = 128;
constexpr int large_block_size = 1024;
constexpr int MAX_LEN = 256;

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    int x = blockIdx.x * k_block_x + threadIdx.y;
    int y = blockIdx.y * k_block_y + threadIdx.x;
    
    if (x >= num_v) return;

    int tid = x * INFEATURE + y;
    int begin = ptr[x], end = ptr[x + 1];
    
    float result = 0.0f;

    for (int i = begin; i < end; i += k_block_y) {
        int local_i = i + threadIdx.x;
        int local_end = min(k_block_y, end - i);

        int s_idx = 0;
        float s_val = 0.0f;
        if (local_i < end) {
            s_idx = idx[local_i];
            s_val = val[local_i];
        }
        
        #pragma unroll(32)
        for (int idx = 0; idx < local_end; idx++) {
            int c = __shfl_sync(0xffffffff, s_idx, idx);
            float v = __shfl_sync(0xffffffff, s_val, idx);
            result += v * vin[c * INFEATURE + y];
        }
    }

    vout[tid] = result;
}

__global__ void spmm_kernel_roma(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    int x = blockIdx.x * k_block_x + threadIdx.y;
    int y = blockIdx.y * k_block_y + threadIdx.x;
    
    if (x >= num_v) return;

    int tid = x * INFEATURE + y;
    int begin = ptr[x], end = ptr[x + 1];
    int before_begin = begin - begin % k_block_y;
    int middle_begin = min(end, before_begin + k_block_y);
    int after_begin = max(middle_begin, end - end % k_block_y);
    
    float result = 0.0f;

    {
        int local_i = before_begin + threadIdx.x;
        int local_end = min(k_block_y, end - before_begin);

        int s_idx = 0;
        float s_val = 0.0f;
        if (local_i >= begin && local_i < end) {
            s_idx = idx[local_i];
            s_val = val[local_i];
        }
        
        #pragma unroll(32)
        for (int idx = 0; idx < local_end; idx++) {
            int c = __shfl_sync(0xffffffff, s_idx, idx);
            float v = __shfl_sync(0xffffffff, s_val, idx);
            result += v * vin[c * INFEATURE + y];
        }
    }

    for (int i = middle_begin; i < after_begin; i += k_block_y) {
        int local_i = i + threadIdx.x;
        int s_idx = idx[local_i];
        float s_val = val[local_i];
        
        #pragma unroll(32)
        for (int idx = 0; idx < k_block_y; idx++) {
            int c = __shfl_sync(0xffffffff, s_idx, idx);
            float v = __shfl_sync(0xffffffff, s_val, idx);
            result += v * vin[c * INFEATURE + y];
        }
    }

    if (after_begin < end) {
        int local_i = after_begin + threadIdx.x;
        int local_end = min(k_block_y, end - after_begin);

        int s_idx = 0;
        float s_val = 0.0f;
        if (local_i < end) {
            s_idx = idx[local_i];
            s_val = val[local_i];
        }
        
        #pragma unroll(32)
        for (int idx = 0; idx < local_end; idx++) {
            int c = __shfl_sync(0xffffffff, s_idx, idx);
            float v = __shfl_sync(0xffffffff, s_val, idx);
            result += v * vin[c * INFEATURE + y];
        }
    }

    vout[tid] = result;
}

__global__ void spmm_kernel_sm(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    int x = blockIdx.x * k_block_x + threadIdx.y;
    int y = blockIdx.y * k_block_y + threadIdx.x;

    if (x >= num_v) return;

    int tid = x * INFEATURE + y;
    int lane_id = threadIdx.x;
    int begin = ptr[x], end = ptr[x + 1];
    int tmp_end = end - (end - begin) % k_block_y;

    float result = 0.0f, result0 = 0.0f, result1 = 0.0f;
    float result2 = 0.0f, result3 = 0.0f;

    __shared__ int s_idx[k_block];
    __shared__ float s_val[k_block];

    int bias = 32 * threadIdx.y;
    for (int i = begin; i < tmp_end; i += k_block_y) {
        int local_i = i + lane_id;

        s_idx[lane_id + bias] = idx[local_i];
        s_val[lane_id + bias] = val[local_i];
        __syncwarp();

        for (int idx = 0; idx < k_block_y; idx += 4) {
            int c0 = s_idx[idx + bias];
            int c1 = s_idx[idx + bias + 1];
            int c2 = s_idx[idx + bias + 2];
            int c3 = s_idx[idx + bias + 3];
            result0 += s_val[idx + bias] * vin[c0 * INFEATURE + y];
            result1 += s_val[idx + bias + 1] * vin[c1 * INFEATURE + y];
            result2 += s_val[idx + bias + 2] * vin[c2 * INFEATURE + y];
            result3 += s_val[idx + bias + 3] * vin[c3 * INFEATURE + y];
        }
    }

    {
        int local_i = tmp_end + lane_id;
        int local_end = end - tmp_end;

        if (local_i < end) {
            s_idx[lane_id + bias] = idx[local_i];
            s_val[lane_id + bias] = val[local_i];
        }
        __syncwarp();
        
        for (int idx = 0; idx < local_end; idx++) {
            int c = s_idx[idx + bias];
            result += s_val[idx + bias] * vin[c * INFEATURE + y];
        }
    }

    vout[tid] = result + result0 + result1 + result2 + result3;
}


__global__ void spmm_large_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE, int *tar_row, int *tar_ptr) {
    int xid = blockIdx.x * blockDim.y + threadIdx.y;
    int y = blockIdx.y * k_block_y + threadIdx.x;

    if (xid >= num_v) return;
    int x = tar_row[xid];

    int tid = x * INFEATURE + y;
    int lane_id = threadIdx.x;
    int begin = tar_ptr[xid], end = min(tar_ptr[xid + 1], ptr[x + 1]);
    int tmp_end = end - (end - begin) % k_block_y;

    float result = 0.0f, result0 = 0.0f, result1 = 0.0f;
    float result2 = 0.0f, result3 = 0.0f;

    __shared__ int s_idx[large_block_size];
    __shared__ float s_val[large_block_size];

    int bias = 32 * threadIdx.y;
    for (int i = begin; i < tmp_end; i += k_block_y) {
        int local_i = i + lane_id;

        s_idx[lane_id + bias] = idx[local_i];
        s_val[lane_id + bias] = val[local_i];
        __syncwarp();

        for (int idx = 0; idx < k_block_y; idx += 4) {
            int c0 = s_idx[idx + bias];
            int c1 = s_idx[idx + bias + 1];
            int c2 = s_idx[idx + bias + 2];
            int c3 = s_idx[idx + bias + 3];
            result0 += s_val[idx + bias] * vin[c0 * INFEATURE + y];
            result1 += s_val[idx + bias + 1] * vin[c1 * INFEATURE + y];
            result2 += s_val[idx + bias + 2] * vin[c2 * INFEATURE + y];
            result3 += s_val[idx + bias + 3] * vin[c3 * INFEATURE + y];
        }
    }

    {
        int local_i = tmp_end + lane_id;
        int local_end = end - tmp_end;

        if (local_i < end) {
            s_idx[lane_id + bias] = idx[local_i];
            s_val[lane_id + bias] = val[local_i];
        }
        __syncwarp();
        
        for (int idx = 0; idx < local_end; idx++) {
            int c = s_idx[idx + bias];
            result += s_val[idx + bias] * vin[c * INFEATURE + y];
        }
    }

    result += result0 + result1 + result2 + result3;
    atomicAdd(&vout[tid], result);
}

__global__ void spmm_kernel_sm256(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE) {
    int x = blockIdx.x * k_block_x + threadIdx.y;
    int y = blockIdx.y * sm256_y + threadIdx.x;

    if (x >= num_v) return;

    int tid = x * INFEATURE + y;
    int lane_id = threadIdx.x;
    int begin = ptr[x], end = ptr[x + 1];
    int tmp_end = end - (end - begin) % k_block_y;

    float result0 = 0.0f, result1 = 0.0f;
    float result2 = 0.0f, result3 = 0.0f;

    __shared__ int s_idx[k_block];
    __shared__ float s_val[k_block];

    int bias = 32 * threadIdx.y;
    for (int i = begin; i < tmp_end; i += k_block_y) {
        int local_i = i + lane_id;

        s_idx[lane_id + bias] = idx[local_i];
        s_val[lane_id + bias] = val[local_i];
        __syncwarp();

        for (int idx = 0; idx < k_block_y; idx++) {
            int c = s_idx[idx + bias] * INFEATURE + y;
            float v = s_val[idx + bias];
            result0 += v * vin[c];
            result1 += v * vin[c + 32];
            result2 += v * vin[c + 64];
            result3 += v * vin[c + 96];
        }
    }

    {
        int local_i = tmp_end + lane_id;
        int local_end = end - tmp_end;

        if (local_i < end) {
            s_idx[lane_id + bias] = idx[local_i];
            s_val[lane_id + bias] = val[local_i];
        }
        __syncwarp();
        
        for (int idx = 0; idx < local_end; idx++) {
            int c = s_idx[idx + bias] * INFEATURE + y;
            float v = s_val[idx + bias];
            result0 += v * vin[c];
            result1 += v * vin[c + 32];
            result2 += v * vin[c + 64];
            result3 += v * vin[c + 96];
        }
    }

    vout[tid] = result0;
    vout[tid + 32] = result1;
    vout[tid + 64] = result2;
    vout[tid + 96] = result3;
}

__global__ void spmm_small_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE, int *row_idx) {
    int x = blockIdx.x * k_block_x + threadIdx.y;
    int y = blockIdx.y * k_block_y + threadIdx.x;

    if (x >= num_v) return;
    x = row_idx[x];

    int tid = x * INFEATURE + y;
    int lane_id = threadIdx.x;
    int begin = ptr[x], end = ptr[x + 1];
    int tmp_end = end - (end - begin) % k_block_y;

    float result = 0.0f, result0 = 0.0f, result1 = 0.0f;
    float result2 = 0.0f, result3 = 0.0f;

    __shared__ int s_idx[k_block];
    __shared__ float s_val[k_block];

    int bias = 32 * threadIdx.y;
    for (int i = begin; i < tmp_end; i += k_block_y) {
        int local_i = i + lane_id;

        s_idx[lane_id + bias] = idx[local_i];
        s_val[lane_id + bias] = val[local_i];
        __syncwarp();

        for (int idx = 0; idx < k_block_y; idx += 4) {
            int c0 = s_idx[idx + bias];
            int c1 = s_idx[idx + bias + 1];
            int c2 = s_idx[idx + bias + 2];
            int c3 = s_idx[idx + bias + 3];
            result0 += s_val[idx + bias] * vin[c0 * INFEATURE + y];
            result1 += s_val[idx + bias + 1] * vin[c1 * INFEATURE + y];
            result2 += s_val[idx + bias + 2] * vin[c2 * INFEATURE + y];
            result3 += s_val[idx + bias + 3] * vin[c3 * INFEATURE + y];
        }
    }

    {
        int local_i = tmp_end + lane_id;
        int local_end = end - tmp_end;

        if (local_i < end) {
            s_idx[lane_id + bias] = idx[local_i];
            s_val[lane_id + bias] = val[local_i];
        }
        __syncwarp();
        
        for (int idx = 0; idx < local_end; idx++) {
            int c = s_idx[idx + bias];
            result += s_val[idx + bias] * vin[c * INFEATURE + y];
        }
    }

    vout[tid] = result + result0 + result1 + result2 + result3;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    pii *rsize = new pii[num_v];
    int zero = 0, cnt = 0;

    int large_x = 16;

    if (feat_in == 256) {
        switch (num_v) {
            case 169343:    // arxiv
            case 1138499:   // youtube
            case 881680:    // am
                mode = 1;
                large_x = 32;
                break;
            case 716847:    // yelp
                mode = 1;
                large_x = 16;
                break;
            case 235868:    // collab
            case 2927963:   // citation
            case 2500604:   // wikikg2
            case 2449029:   // products
            case 4267:      // ddi
                mode = 2;
                break;
            case 576289:    // ppa
            case 1569960:   // amazon_cogdl
                mode = 4;
                break;
            case 232965:    // reddit.dgl
                mode = 5;
                break;
            case 132534:    // protein
                mode = 6;
                break;
            default:
                mode = 1;
                break;
        }
    } else {
        switch (num_v) {
            case 2449029:   // products
                mode = 0;
                break;
            case 169343:    // arxiv
            case 1138499:   // youtube
            case 881680:    // am
                mode = 1;
                large_x = 32;
                break;
            case 716847:    // yelp
            case 2927963:   // citation
                mode = 1;
                large_x = 16;
                break;
            case 4267:      // ddi
            case 235868:    // collab
                mode = 1;
                large_x = 8;
                break;
            case 132534:    // protein
                mode = 1;
                large_x = 4;
                break;
            case 2500604:   // wikikg2
                mode = 3;
                break;
            case 576289:    // ppa
            case 1569960:   // amazon_cogdl
                mode = 4;
                break;  
            case 232965:    // reddit.dgl
                mode = 5;
                break;
            default:
                mode = 1;
                break;
        }
    }

    if (mode == 2 && feat_in != 256) mode = 1;

    int *ptr = new int[num_v + 1];
    int *l_idx = new int[num_e];
    float *l_val = new float[num_e];
    checkCudaErrors(cudaMemcpy(ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(l_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(l_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost));

    // merge duplicate
    int *tmp_ptr = new int[num_v + 1];
    int *tmp_idx = new int[num_e];
    float *tmp_val = new float[num_e];

    int tmp_end = 0;
    for (int i = 0; i < num_v; i++) {
        int begin = ptr[i], end = ptr[i + 1];
        int tmp_begin = tmp_end;
        int last_idx = -1;
        for (int j = begin; j < end; j++) {
            if (l_val[j] == 0) continue;
            if (l_idx[j] != last_idx) {
                tmp_idx[tmp_end] = l_idx[j];
                tmp_val[tmp_end] = l_val[j];
                tmp_end++;
                last_idx = l_idx[j];
            } else {
                tmp_val[tmp_end - 1] += l_val[j];
            }
        }
        tmp_ptr[i] = tmp_begin;
        tmp_ptr[i + 1] = tmp_end;
    }

    num_e = tmp_end;

    memcpy(ptr, tmp_ptr, (num_v + 1) * sizeof(int));
    memcpy(l_idx, tmp_idx, num_e * sizeof(int));
    memcpy(l_val, tmp_val, num_e * sizeof(float));

    checkCudaErrors(cudaMalloc2((void**)&new_ptr, (num_v + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&new_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&new_val, num_e * sizeof(float)));

    checkCudaErrors(cudaMemcpy(new_ptr, tmp_ptr, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(new_idx, tmp_idx, num_e * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(new_val, tmp_val, num_e * sizeof(float), cudaMemcpyHostToDevice));

    int *idx = new int[num_v];
    int *nnz = new int[num_v];
    for (int i = 0; i < num_v; i++) {
        nnz[i] = (32 - (ptr[i + 1] - ptr[i]) % 32) % 32;
        cnt += nnz[i];
    }

    int *pad_ptr = new int[num_v + 1];
    int *pad_idx = new int[num_e + cnt];
    float *pad_val = new float[num_e + cnt];

    if (mode == 5) {
        pad_ptr[0] = ptr[0];
        for (int i = 0; i < num_v; i++) {
            int tmp = ptr[i + 1] - ptr[i];
            for (int j = 0; j < tmp; j++) {
                pad_idx[pad_ptr[i] + j] = l_idx[ptr[i] + j];
                pad_val[pad_ptr[i] + j] = l_val[ptr[i] + j];
            }
            pad_ptr[i + 1] = pad_ptr[i] + tmp + nnz[i];
        }
        num_e += cnt;
    }

    checkCudaErrors(cudaMalloc2((void**)&pd_ptr, (num_v + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&pd_idx, num_e * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&pd_val, num_e * sizeof(float)));
    checkCudaErrors(cudaMemcpy(pd_ptr, pad_ptr, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pd_idx, pad_idx, num_e * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pd_val, pad_val, num_e * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_v; i++) {
        if (mode == 5) {
            rsize[i] = std::make_pair(pad_ptr[i + 1] - pad_ptr[i], i);
        } else {
            rsize[i] = std::make_pair(ptr[i + 1] - ptr[i], i);
        }
    }

    if (mode == 4 || mode == 5) {
        // For ppa, reddit, amazon, 10%-30% up
        std::sort(rsize, rsize + num_v, std::greater<pii>());
        for (int i = 0; i < num_v; i++) {
            if (rsize[i].first == 0) zero++;
            idx[i] = rsize[i].second;
        }
        num_v = num_v - zero;
    }

    checkCudaErrors(cudaMalloc2((void**)&row_index, num_v * sizeof(int)));
    checkCudaErrors(cudaMemcpy(row_index, idx, num_v * sizeof(int), cudaMemcpyHostToDevice));

    int *large_row = new int[num_v * large_x];
    int *small_row = new int[num_v];
    int *large_ptr = new int[num_v * large_x + 1];
    lr_cnt = 0;
    sr_cnt = 0;

    if (mode == 1) {
        for (int i = 0; i < num_v; i++) {
            int tmp = ptr[i + 1] - ptr[i];
            if (tmp >= MAX_LEN) {
                // large
                int blk_need = 1;
                if (tmp > 1024) blk_need = 2;
                if (tmp > 4096) blk_need = 3;
                if (tmp > 16384) blk_need = 4;
                if (tmp > 65536) blk_need = 5;
                int blk_cnt = blk_need * large_x;
                int part_len = (tmp + blk_cnt - 1) / blk_cnt;
                large_ptr[lr_cnt] = ptr[i];
                large_row[lr_cnt] = i;
                for (int j = 1; j < blk_cnt; j++) {
                    large_ptr[lr_cnt + j] = large_ptr[lr_cnt + j - 1] + part_len;
                    large_row[lr_cnt + j] = i;
                }

                // temp, not used
                large_ptr[lr_cnt + blk_cnt] = ptr[i + 1];
                lr_cnt += blk_cnt;
            } else {
                // small
                small_row[sr_cnt++] = i;
            }
        }
        num_v = sr_cnt;
    }

    checkCudaErrors(cudaMalloc2((void**)&large_row_index, lr_cnt * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&small_row_index, sr_cnt * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&lar_ptr, (lr_cnt + 1) * sizeof(int)));
    checkCudaErrors(cudaMemcpy(large_row_index, large_row, lr_cnt * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(small_row_index, small_row, sr_cnt * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(lar_ptr, large_ptr, (lr_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    delete[] large_ptr;
    delete[] large_row;
    delete[] small_row;
    delete[] l_idx;
    delete[] pad_ptr;
    delete[] pad_idx;
    delete[] pad_val;
    delete[] rsize;
    delete[] idx;
    delete[] tmp_ptr;
    delete[] tmp_idx;
    delete[] tmp_val;

    grid.x = (num_v + k_block_x - 1) / k_block_x;
    grid.y = (feat_in + k_block_y - 1) / k_block_y;
    // Same warp handles same row
    block.x = k_block_y;
    block.y = k_block_x;

    if (mode == 1) {       
        large_grid.x = (lr_cnt + large_x - 1) / k_block_x;
        large_grid.y = (feat_in + k_block_y - 1) / k_block_y;
        large_block.x = k_block_y;
        large_block.y = large_x;
    }
    
    if (mode == 2) {
        grid.y = 2;
    }

}

void SpMMOpt::run(float *vin, float *vout) {
    switch (mode) {
        case 1:
            spmm_large_kernel<<<large_grid, large_block>>>(new_ptr, new_idx, new_val, vin, vout, lr_cnt, feat_in, large_row_index, lar_ptr);
            spmm_small_kernel<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, sr_cnt, feat_in, small_row_index);
            break;
        case 2:
            spmm_kernel_sm256<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, num_v, feat_in);
            break;
        case 3:
            spmm_kernel_sm<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, num_v, feat_in);
            break;
        case 4:
            spmm_small_kernel<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, num_v, feat_in, row_index);
            break;
        case 5:
            spmm_small_kernel<<<grid, block>>>(pd_ptr, pd_idx, pd_val, vin, vout, num_v, feat_in, row_index);
            break;
        case 6:
            spmm_kernel_roma<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, num_v, feat_in);
            break;
        case 0:
        default:
            spmm_kernel<<<grid, block>>>(new_ptr, new_idx, new_val, vin, vout, num_v, feat_in);
            break;
    }  
}