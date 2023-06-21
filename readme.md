# Cuda-SPMM



## 简介

* 一个利用 CUDA 实现的稀疏矩阵-矩阵乘法（SpMM）算法。
* 主要利用的优化方案包括 CRC, tiling, CWM, Row Swizzle 等。



## 使用方法

* 生成执行文件：

  ```shell
  spack load cuda
  mkdir build
  cd build
  cmake ..
  make -j4
  ```

* 测试：（请在 `./build/` 目录下执行）

  ```shell
  # 运行单个数据点
  srun -p gpu ./test/unit_tests --dataset <datasetname> --len 32 --datadir ../data/
  
  # 运行全部 GPU 数据点
  srun -p gpu ../script/run_all.sh
  
  # 运行全部 CPU 数据点
  srun -p gpu ../script/run_all_CPU.sh
  
  # 也可以改变环境变量，仅仅运行单个测试
  # 如验证正确性（Validation）
  GTEST_FILTER="SpMMTest.validation" srun -p gpu ./test/unit_tests --dataset <datasetname> --len 32 --datadir ../data/
  
  # 或是只运行 CPU 的相关测试
  GTEST_FILTER="SpMMCPUTest.*" srun -p gpu ./test/unit_tests --dataset <datasetname> --len 32 --datadir ../data/
  ```



## 实现方法

* 考虑到数据的稀疏情况有所不同，对于非零元较多的行，和一个线程完成一个值（$1\times1$）的实现方式相比，运算时间会显著提升，注意到每个线程块需要等待其中的所有线程完成工作后才能释放资源，因此不应为单一线程分配过多的任务。



### 1. 矩阵分片：行合并访存（CRC）方法

* 为了解决 Warp divergence 的问题，一种解决方法是把一个线程块映射到稀疏矩阵的几行，每个 Warp 负责计算一行数据，而 Warp 中的 32 个线程分別对应稠密矩阵的 32 列（[Yang et al., 2018](#参考资料)），为实现上的简便，当 K=256 时，我们分配 8 个线程块来分別计算矩阵的 $[32\times bid, 32\times (bid+1))$ 列。

  

### 2. 负载不均衡：分层 tiling 方法

* 某些数据集中同时存在大量的稀疏行或稠密行。在这种情况下，如果对于所有行我们都一视同仁分配 32 个线程来计算的话，会导致较明显的负载不均衡现象。

* 因此，我们应当分配更多的线程来计算这些较 “稠密” 的行，一种方式是把所有的行切分为相同大小的 tile，然后每个线程计算这个 tile 的数据和稠密矩阵一列的结果。（[Gale et al., 2020](#参考资料)）

* 不过为了实现上的简便，我采用了一种替代方法：稠密行（非零元个数大于 256 的行）采用新的 Kernel 计算，此时每个线程块中包含 $32\times32$ 个线程（用于把分配到的非零元平均划分为 32 段计算），每行按照数据量划分为 1 到 5 个线程块，由于一行被切分到多个 Warp 中计算，所以需要利用原子加操作统整结果。对于稀疏行，则仍然使用原来的 Kernel 计算。



### 3. 矩阵分片：粗粒度 Warp 合并（CWM）方法

* 某些数据集中仅有 0.1% 的行有多于 30 个非零元，对于包含最多非零元的一行，其非零元个数不超过一千个。对于如此稀疏的矩阵，在计算每一行的 256 列时分为 8 个线程块的话会造成大量资源浪费，因此在一个线程中计算的值由原本的一个（$1\times1$）改为了四个（$1\times4$），可以有效减少线程块的数量，取得更好的效能。（[Huang et al., 2020](#参考资料)）

  

### 4. 负载不均衡：Row Swizzle 负载均衡方法

* 某些数据集中存在不少的 “稠密行” （行中非零元个数大于 3%）。为此，我们可以通过重新映射行的方式来把计算量相近的行放在相邻的 Warp 中，使得 SM 和 Warp 都满足负载均衡的条件。

* 方法十分简单，按照行中非零元的个数由大至小排列并依次映射到 Warp 中即可。（[Gale et al., 2020](#参考资料)）



## 参考资料

```
@misc{yang2018design,
      title={Design Principles for Sparse Matrix Multiplication on the GPU}, 
      author={Carl Yang and Aydin Buluc and John D. Owens},
      year={2018},
      eprint={1803.08601},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

```
@misc{huang2020gespmm,
      title={GE-SpMM: General-purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks}, 
      author={Guyue Huang and Guohao Dai and Yu Wang and Huazhong Yang},
      year={2020},
      eprint={2007.03179},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

```
@INPROCEEDINGS{9355309,
      author={Gale, Trevor and Zaharia, Matei and Young, Cliff and Elsen, Erich},
      booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis}, 
      title={Sparse GPU Kernels for Deep Learning}, 
      year={2020},
      volume={},
      number={},
      pages={1-14},
      doi={10.1109/SC41405.2020.00021}
}
```
