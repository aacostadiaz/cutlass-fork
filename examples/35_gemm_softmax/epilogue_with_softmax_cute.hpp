/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_
>
class EpilogueWithSoftmax {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;

  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    float* ptr_Sum = nullptr;
    float* ptr_Max = nullptr;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueWithSoftmax(Params const& params_)
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    using Minus = minus<float>;
    using Exp   = fast_exp_op<float>;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if CUDA_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
#else
    auto synchronize = [] () { syncthreads(); };
#endif

    Minus     minus;
    Exp       exponential;

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    auto SN = ceil_div(N, get<1>(blk_shape_MNK));

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);      //             (m,n,l)
    Tensor mS_mnl = make_tensor(make_gmem_ptr(params.ptr_Sum), make_shape(M, SN, L), make_stride(1, M, M*SN));
    Tensor mN_mnl = make_tensor(make_gmem_ptr(params.ptr_Max), make_shape(M, SN, L), make_stride(1, M, M*SN));
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N, m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N, m,n,l)
    Tensor gS_mnl = local_tile(mS_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1, X, X>{});      // (BLK_M,m,sn,l)
    Tensor gN_mnl = local_tile(mN_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1, X, X>{});      // (BLK_M,m,sn,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gS = gS_mnl(_,m_coord,n_coord,l_coord);                                                     // (BLK_M)
    Tensor gN = gN_mnl(_,m_coord,n_coord,l_coord);                                                     // (BLK_M)

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});              // (SMEM_M,SMEM_N)

    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto tC     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto tD     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrC));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    auto thr_block = make_layout(typename TiledCopyS2R::TiledNumThr{});
    Tensor tSgS = local_partition(gS, thr_block, ThreadIdxX());         // (THR_M)
    Tensor tNgN = local_partition(gN, thr_block, ThreadIdxX());         // (THR_M)

    Tensor tSrS = make_tensor<float>(shape(tSgS));
    Tensor tNrN = make_tensor<float>(shape(tNgN));

    // Predication
    Tensor tSpS = make_tensor<bool>(shape(tSrS));

    uint32_t float_max_bits = 0xff7fffff;
    float min_float = reinterpret_cast<float const &>(float_max_bits);
    fill(tNrN, min_float);
    clear(tSrS);

    CUTE_STATIC_ASSERT(get<0>(blk_shape_MNK) % typename TiledCopyS2R::TiledNumThr{} == 0);
    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N
    CUTE_STATIC_ASSERT(typename TiledCopyS2R::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));

#if 0
    if (thread_idx == 0 && m_coord == 0 && n_coord == 0 && l_coord == 0) {
      print("mC_mnl   : "); print(mC_mnl.layout()); print("\n");
      print("mD_mnl   : "); print(mD_mnl.layout()); print("\n");
      print("\n");
      print("aC   : "); print(accumulators.layout()); print("\n");
      print("gC   : "); print(gC.layout()); print("\n");
      print("gD   : "); print(gD.layout()); print("\n");
      print("sC   : "); print(sC.layout()); print("\n");
      print("\n");
      print("tCsC : "); print(tCsC.layout()); print("\n");
      print("tCaC : "); print(tCaC.layout()); print("\n");
      print("\n");
      print("gDt  : "); print(gDt.layout()); print("\n");
      print("tDsC : "); print(tDsC.layout()); print("\n");
      print("tDrC : "); print(tDrC.layout()); print("\n");
      print("\n");
      print("tDrD : "); print(tDrD.layout()); print("\n");
      print("tDgC : "); print(tDgC.layout()); print("\n");
      print("tDgD : "); print(tDgD.layout()); print("\n");
      print("\n");
      print("mS_mnl : "); print(mS_mnl.layout()); print("\n");
      print("mN_mnl : "); print(mN_mnl.layout()); print("\n");
      print("gS_mnl : "); print(gS_mnl.layout()); print("\n");
      print("gN_mnl : "); print(gN_mnl.layout()); print("\n");
      print("gS : "); print(gS.layout()); print("\n");
      print("gN : "); print(gN.layout()); print("\n");
      print("\n");
      print("tSgS : "); print(tSgS.layout()); print("\n");
      print("tNgN : "); print(tNgN.layout()); print("\n");
      print("tSrS : "); print(tSrS.layout()); print("\n");
      print("tNrN : "); print(tNrN.layout()); print("\n");
      print("\n");
    }
#endif

    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n)
      {
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;

            copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
          }
        }

        // Step 2. Wait for SMEM writes to complete
        synchronize();

        // Step 3. Copy from SMEM into a fragment
        copy(tiled_s2r, tDsC, tDrC);

        // Step 4. Wait for SMEM reads to complete
        synchronize();

        Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
        Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

        if (epilogue_op.is_source_needed()) {
          // source is needed
          Tensor tDgCmn = tDgC(_,_,_,step_m,step_n);
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m)
          {
            auto row_coord = step_m * size<2>(cDt) + m;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n)
            {
              // Predication
              tSpS(row_coord) = get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk);
              if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                  get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk))
              {
                // Step 5. Elementwise operation with conversion
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size<0>(tDrC); ++i) {
                  tDrD(i,m,n) = epilogue_op(tDrC(i,m,n), tDgCmn(i,m,n));
                }

                float max_acc_prev = tNrN(row_coord);
                for (int v = 0; v < size<0>(tDrD); ++v) {
                  tNrN(row_coord) = fast_max(tNrN(row_coord), static_cast<float>(tDrD(v,m,n)));
                }

                float max_acc = tNrN(row_coord);
                if (step_n != 0 || n != 0) {
                  tSrS(row_coord) *= fast_exp(max_acc_prev - max_acc);
                }
                for (int v = 0; v < size<0>(tDrD); ++v) {
                  tSrS(row_coord) += exponential(minus(static_cast<float>(tDrD(v,m,n)), max_acc));
                }

                // Step 6. Copy to GMEM
                copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
              }
            }
          }
        }
        else {
          // source is not needed, avoid load and lift compute

          // Step 5. Elementwise operation with conversion
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tDrC); ++i) {
            tDrD(i) = epilogue_op(tDrC(i));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m)
          {
            auto row_coord = step_m * size<2>(cDt) + m;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n)
            {
              // Predication
              tSpS(row_coord) = get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk);
              if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                  get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk))
              {
                float max_acc_prev = tNrN(row_coord);
                for (int v = 0; v < size<0>(tDrD); ++v) {
                  tNrN(row_coord) = fast_max(tNrN(row_coord), static_cast<float>(tDrD(v,m,n)));
                }

                float max_acc = tNrN(row_coord);
                if (step_n != 0 || n != 0) {
                  tSrS(row_coord) *= fast_exp(max_acc_prev - max_acc);
                }
                for (int v = 0; v < size<0>(tDrD); ++v) {
                  tSrS(row_coord) += exponential(minus(static_cast<float>(tDrD(v,m,n)), max_acc));
                }

                // Step 6. Copy to GMEM
                copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
              }
            }
          }
        }
      }
    }
    copy_if(tSpS, tSrS, tSgS);
    copy_if(tSpS, tNrN, tNgN);
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
