#include "gtest/gtest.h"
#include "util.h"
#include "valid.h"
#include "spmm_cpu_ref.h"
#include "spmm_cpu_opt.h"

class SpMMCPUTest : public testing::Test
{
protected:
    vector<void *> tensor_ptr;
    float *p_in_feat_vec, *p_out_feat_vec, *p_out_feat_vec_ref, *p_value;
    CSR *g;
    virtual void SetUp()
    {
        p_in_feat_vec = allocate<float>(kNumV * kLen, &tensor_ptr, true, 1);
        p_out_feat_vec = allocate<float>(kNumV * kLen, &tensor_ptr, false, 1);
        p_out_feat_vec_ref = allocate<float>(kNumV * kLen, &tensor_ptr, false, 1);
        p_value = allocate<float>(kNumE, &tensor_ptr, true, 1);
        g = new CSR(kNumV, kNumE, gptr_cpu, gidx_cpu, p_value);
    }
    virtual void TearDown()
    {
        for (auto item : tensor_ptr)
        {
            delete[] (float*)item;
        }
    }
};

TEST_F(SpMMCPUTest, validation)
{
    SpMMCPURef *spmmer_ref = new SpMMCPURef(g, kLen);
    SpMMCPUOpt *spmmer = new SpMMCPUOpt(g, kLen);
    spmmer_ref->preprocess(p_in_feat_vec, p_out_feat_vec_ref);
    spmmer->preprocess(p_in_feat_vec, p_out_feat_vec);
    memset(p_out_feat_vec, 0, sizeof(float) * kNumV * kLen);
    memset(p_out_feat_vec_ref, 0, sizeof(float) * kNumV * kLen);
    spmmer_ref->run(p_in_feat_vec, p_out_feat_vec_ref);
    spmmer->run(p_in_feat_vec, p_out_feat_vec);
    ASSERT_LT(valid_cpu(p_out_feat_vec, p_out_feat_vec_ref, kNumV * kLen), kNumV * kLen / 10000 + 1);
}


TEST_F(SpMMCPUTest, opt_performance)
{
    SpMMCPUOpt *spmmer = new SpMMCPUOpt(g, kLen);
    spmmer->preprocess(p_in_feat_vec, p_out_feat_vec);
    auto time = getAverageTimeWithWarmUp([&]()
                                         { spmmer->run(p_in_feat_vec, p_out_feat_vec); });
    dbg(time);
}
