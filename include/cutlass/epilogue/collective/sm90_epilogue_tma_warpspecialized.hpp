/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  class CtaTileMNK_,   //     (CTA_M,CTA_N,CTA_K)
  class EpilogueTile_, // (EPI_TILE_M,EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyAtomC_,
  class CopyOpR2R_
>
class CollectiveEpilogue<
    Sm90TmaWarpSpecialized<StagesC_,StagesD_,FragmentSize_,ReuseSmemC_,DelayTmaStore_>,
    CtaTileMNK_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyAtomC_,
    CopyOpR2R_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Sm90TmaWarpSpecialized<StagesC_,StagesD_,FragmentSize_,ReuseSmemC_,DelayTmaStore_>;
  using CtaTileMNK = CtaTileMNK_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyAtomC = CopyAtomC_;
  using CopyOpR2R = CopyOpR2R_;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
  static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, "EPI_TILE_M must divide CTA_M");
  static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
  using NonVoidElementD = cute::conditional_t<not is_destination_supported,fusion::get_element_aux_t<FusionCallbacks>, ElementD>;
  static_assert(not cute::is_void_v<NonVoidElementD>, "SmemElementD is void");
  using NonVoidElementC = cute::conditional_t<not is_source_supported,NonVoidElementD,ElementC>; // prevents void ref breakages

  using TmaElementD = cute::conditional_t<cute::is_same_v<NonVoidElementD, cutlass::complex<float>>, uint64_t, NonVoidElementD>;
  using TmaElementC = cute::conditional_t<cute::is_same_v<NonVoidElementC, cutlass::complex<float>>, uint64_t, NonVoidElementC>;

  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;

  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
  constexpr static bool DelayTmaStore = DelayTmaStore_;

  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();

  constexpr static bool is_im2col_C = cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>;
  constexpr static bool is_im2col_D = cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>;

  // Check if register transformation is needed before copying register to shared memory.
  constexpr static bool IsUseR2R = !cute::is_void_v<CopyOpR2R>;

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesC>{}),
      cute::conditional_t<is_m_major_C, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<ReuseSmemC ? StagesC : StagesD>{}),
      cute::conditional_t<is_m_major_D, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));

  constexpr static bool support_smem_reuse = is_source_supported && is_destination_supported && StagesD <= StagesC
                                            && cosize(take<0,2>(SmemLayoutC{})) == cosize(take<0,2>(SmemLayoutD{}));
  static_assert(not (ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");

  constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});
  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  using SmemArrayTypeC = cute::ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>>;
  using SmemArrayTypeD = cute::ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>>;

  using EmptyType = cute::tuple<>;
  using SmemCStorage = cute::conditional_t<is_source_supported and (not ReuseSmemC),
                         SmemArrayTypeC,
                         EmptyType>;
  using SmemDStorage = cute::conditional_t<is_destination_supported,
                         SmemArrayTypeD,
                         EmptyType>;

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using LoadPipelineState = cutlass::PipelineState<StagesC>;
  constexpr static uint32_t TmaTransactionBytes =
    (size(take<0,2>(SmemLayoutC{})) * static_cast<uint32_t>(sizeof_bits<SmemElementC>::value)) / 8;
  constexpr static bool RequiresTransactionBytes = true;

  // TMA pipeline for storing D
  using StorePipeline = cute::conditional_t<ReuseSmemC,
                          cutlass::PipelineTmaStore<StagesC, StagesD-1>,
                          cutlass::PipelineTmaStore<StagesD>>;
  using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    } tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD const* ptr_D;
    StrideD dD;
  };

  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(make_tma_copy(
        CopyOpG2S{},
        make_tensor(make_gmem_ptr<TmaElementC const>(nullptr),
            repeat_like(StrideC{}, int32_t(0)), StrideC{}),
        take<0,2>(SmemLayoutC{}),
        EpilogueTile{},
        _1{}));
    using TMA_D = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(make_gmem_ptr<TmaElementD>(nullptr),
            repeat_like(StrideD{}, int32_t(0)), StrideD{}),
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{},
        _1{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_D tma_store_d;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    uint32_t transaction_bytes = TmaTransactionBytes;
    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
      Tensor tensor_c = make_tensor(make_gmem_ptr<TmaElementC const>(args.ptr_C), make_layout(make_shape(M,N,L), args.dC));
      tma_load_c = make_tma_copy_C_sm90(
          CopyOpG2S{},
          tensor_c,
          take<0,2>(SmemLayoutC{}),
          EpilogueTile{});
    }

    typename Params::TMA_D tma_store_d;
    if constexpr (is_destination_supported) {
      Tensor tensor_d = make_tensor(make_gmem_ptr<TmaElementD>(args.ptr_D), make_layout(make_shape(M,N,L), args.dD));
      tma_store_d = make_tma_copy_C_sm90(
          CopyOpS2G{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{});
    }

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      tma_load_c,
      tma_store_d,
      transaction_bytes
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    auto shape = cute::make_shape(M,N,L);

    bool implementable = true;
    if constexpr (is_destination_supported) {
      constexpr int tma_alignment_bits_D = cutlass::detail::get_output_alignment_bits<ElementD>();
      constexpr int min_tma_aligned_elements_D = tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
      if constexpr (cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>) { // ignore L stride for implicit gemm
        implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_D>(take<0,2>(shape), take<0,2>(StrideD{}));
      }
      else {
        implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_D>(shape, StrideD{});
      }
    }

    if constexpr (not cute::is_void_v<ElementC>) {
      constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
      constexpr int min_tma_aligned_elements_C = tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
      if constexpr (cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>) { // ignore L stride for implicit gemm
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(take<0,2>(shape), take<0,2>(StrideC{}));
      }
      else {
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(shape, StrideC{});
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    bool fusion_implementable = FusionCallbacks::can_implement(problem_shape, args.thread);

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    bool beta_implementable = true;

    if constexpr (cute::is_void_v<ElementC>) {
      if constexpr (detail::has_beta<Arguments>::value) {
        beta_implementable = args.thread.beta == 0.0;
      }
      if constexpr (detail::has_beta_ptr<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr == nullptr;
      }
    }

    if (!beta_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Beta/beta pointer was set, but epilogue is sourceless (void-C).\n");
    }

    return implementable && fusion_implementable && beta_implementable;
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment(TileShapeMNK tile_shape_MNK) {
    // Compute number of epilogue subtiles
    return size<1>(zipped_divide(make_layout(take<0,2>(tile_shape_MNK)), EpilogueTile{}));
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment(TileShapeMNK tile_shape_MNK) {
    return get_load_pipe_increment(tile_shape_MNK);
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void
  prefetch_tma_descriptors(Params const& epilogue_params) {
    if constexpr (is_source_supported) {
      cute::prefetch_tma_descriptor(epilogue_params.tma_load_c.get_tma_descriptor());
    }
    if constexpr (is_destination_supported) {
      cute::prefetch_tma_descriptor(epilogue_params.tma_store_d.get_tma_descriptor());
    }
  }

  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      int subtile_idx=-1) {
    using namespace cute;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    // The tma tensor C under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape = conditional_return<is_im2col_C>(
      make_coord(m_coord, n_coord),
      make_coord(m_coord, n_coord, l_coord));

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(make_shape(M,N,L));                                //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (G2S,G2S_M,G2S_N,PIPE_C)

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs(
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      thread_idx
                    );
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gC_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gC_epi); ++epi_m) {
        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gC_epi)) + epi_m) != subtile_idx) {
          continue;
        }
        // Acquire the lock for this stage
        constexpr uint16_t mcast_mask = 0;
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
        load_pipeline.producer_acquire(load_pipe_producer_state);

        // Loop fusion callback entry point
        pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), issue_tma_load);

        // Execute the TMA load for C if needed
        if (issue_tma_load && is_C_load_needed) {
          copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
              bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
          load_pipeline.producer_expect_transaction(load_pipe_producer_state);
        }

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state) {
    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      load_pipeline.producer_tail(load_pipe_producer_state);
    }

    return load_pipe_producer_state;
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      int subtile_idx=-1) {
    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be RF resident.");
    static_assert(rank(AccLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<TileShapeMNK>::value, "TileShapeMNK must be static");
    static_assert(rank(TileShapeMNK{}) == 3, "TileShapeMNK must be rank 3");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape = conditional_return<is_im2col_D>( 
        make_coord(m_coord, n_coord),
        make_coord(m_coord, n_coord, l_coord));

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mD_mn = params.tma_store_d.get_tma_tensor(make_shape(M,N,L));                               //       (M,N,L)
    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtiling
    Tensor gD_epi = flat_divide(gD, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)

    TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

    // (t)hread-partition for (r)egister to (r)egister copy (tRR_)
    TiledCopy tiled_r2r = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsUseR2R) {
        return make_tiled_copy_S(Copy_Atom<CopyOpR2R, ElementCompute>{}, tiled_copy_C_atom);
      }
      else {
        return make_tiled_copy_S(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
          ElementCompute>{}, tiled_copy_C_atom);
      }
    }();
    ThrCopy thread_r2r = tiled_r2r.get_slice(thread_idx);

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsUseR2R) {
        return make_tiled_copy_D(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_r2r);
      }
      else {
        return make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);
      }
    }();
    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);                                   // ((R2S,R2S_V),MMA_M,MMA_N)
    Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);                                       // (R2S,R2S_M,R2S_N,PIPE_D)

    auto mma_tile_m = size<0>(TileShapeMNK{}) / size<1>(tRS_rAcc);
    auto mma_tile_n = size<1>(TileShapeMNK{}) / size<2>(tRS_rAcc);
    auto epi_tile_m = size<0>(EpilogueTile{});
    auto epi_tile_n = size<1>(EpilogueTile{});

    // Allocate D registers
    Layout tRS_rD_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sD_epi))));
    Tensor tRS_rD = make_tensor<SmemElementD>(tRS_rD_layout);                                      // (R2S,R2S_M,R2S_N)

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    Tensor tRS_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(tRS_rAcc);
    Tensor tRS_rD_frg   = recast<Array<SmemElementD      , FragmentSize>>(tRS_rD);
    CUTE_STATIC_ASSERT(size<0>(tRS_rAcc) % FragmentSize == 0, "Fragment size does not vectorize properly");

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_copy_C_atom);
    ThrCopy thread_s2r = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Layout tSR_rC_layout = thread_s2r.retile_D(tRS_rD).layout();                            // (S2R,S2R_M,S2R_N)

    // Allocate C registers
    // If C smem load is a non-vectorized dst(i) = src(i) then we can allocate C registers directly in the compute type
    // to eliminate some redundant pack+unpack instruction sequences for sub-word types
    constexpr bool IsDirectS2R = cute::is_same_v<CopyOpS2R, AutoVectorizingCopyWithAssumedAlignment<128>>
                                && decltype(max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
    using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, SmemElementC>;
    Tensor tRS_rC = make_tensor<RegisterElementC>(tRS_rD_layout);                                  // (R2S,R2S_M,R2S_N)
    Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);                                                   // (S2R,S2R_M,S2R_N)

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsUseR2R) {
        // (t)hread-partition for ConsumerStoreCallbacks. 
        TiledCopy tiled_cst = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementC>{}, tiled_copy_C_atom);
        ThrCopy thread_cst = tiled_cst.get_slice(thread_idx);

        return thread_cst.partition_S(flat_divide(cD_mn, EpilogueTile{}));             // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
      }
      else {
        return thread_r2s.partition_S(flat_divide(cD_mn, EpilogueTile{}));             // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
      }
    }();
    // Relative coordinate tensors (static)
    Tensor cD = make_counting_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());                          // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tRS_cD = make_coord(M,N) - tRS_cD_mn(_0{});                                                   // (m,n)

    CUTE_STATIC_ASSERT(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");

    CUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "EPI_TILE_N must divide MMA_TILE_N");
    // Get TiledCopy for partition reference when consumer store.
    TiledCopy tiled_copy_partition_ref = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);
    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = true; // Register tensors reference tiled copy src layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_copy_partition_ref,
                      cD,
                      residue_cD,
                      tRS_cD,
                      residue_tRS_cD,
                      tRS_rC,
                      thread_idx
                    );
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    bool is_producer_load_needed = fusion_callbacks.is_producer_load_needed();
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    using FragmentVisit = decltype(cst_callbacks.visit(tRS_rAcc_frg(0), 0, 0, 0));
    constexpr bool IsDirectR2S = cute::is_same_v<FragmentVisit, Array<SmemElementD, FragmentSize>>;
    using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
    Tensor tRS_rCompute = make_tensor<RegisterElementD>(tRS_rD_layout);                            // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute_frg = recast<Array<RegisterElementD, FragmentSize>>(tRS_rCompute);

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [&] () CUTLASS_LAMBDA_FUNC_INLINE { cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for TMA store (one warp issues TMA store)
    bool issue_tma_store = (thread_idx / NumThreadsPerWarp) == 0;

    // In the reuse smem configuration we have StagesC smem buffers and at most StagesD committed TMA stores in flight.
    // The TMA store pipeline producer acquire returns when at most StagesD-1 committed stores are in-flight, so we can
    // only guarantee store completion after StagesD iterations, then we can begin issuing releases on the smem buffer locks.
    // store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the release, in circular buffer fashion.
    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }

    // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
    // Sync requirements of smem reuse may preclude this optimization
    // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
    [[maybe_unused]] int epi_m_prev = 0;
    [[maybe_unused]] int epi_n_prev = 0;
    static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The TMA store sequence for one subtile iteration
    auto tma_store_fn = [&] (int epi_m, int epi_n) CUTLASS_LAMBDA_FUNC_INLINE {
      // Write the tile from smem to gmem with TMA
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      synchronize(); // ensure all threads have issued their async fence
      if constexpr (is_destination_supported) {
        if (issue_tma_store) {
          copy(params.tma_store_d, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
        }
      }

      // Post async fence, pre TMA commit callback entry point
      cst_callbacks.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);

      // Commit the TMA stores for this stage
      if (issue_tma_store) {
        store_pipeline.producer_commit(store_pipe_producer_state);
      }
      ++store_pipe_producer_state;
      ++issued_stores;

      // Wait for the next smem buffer to be available
      if (issue_tma_store) {
        store_pipeline.producer_acquire(store_pipe_producer_state);
      }
      synchronize();

      if constexpr (ReuseSmemC) {
        // producer_acquire returns when at most StagesD-1 committed stores are pending
        bool store_finished = issued_stores > StorePipeline::UnacquiredStages;
        // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
        if (store_finished) {
          if (is_producer_load_needed) {
            load_pipeline.consumer_release(load_pipe_consumer_state);
          }
          ++load_pipe_consumer_state;
        }
      }
    };

    //
    // BEGIN EPILOGUE
    //

    // Pre-loop fusion callback entry point
    cst_callbacks.begin();
    if (cst_callbacks.begin_sync_needed()) {
      synchronize();
    }

    // For each output tile
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
        [[maybe_unused]] bool is_first_iteration = epi_m == 0 && epi_n == 0;
        bool is_last_iteration = epi_m == size<2>(gD_epi)-1 && epi_n == size<3>(gD_epi)-1;

        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gD_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        cst_callbacks.begin_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Wait for the producer load to fill smem
          load_pipeline.consumer_wait(load_wait_state);

          if (is_C_load_needed) {
            // Copy source tile from smem to register
            copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
          }
        }

        // First loop fusion callback entry point
        cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

        if (is_producer_load_needed) {
          if constexpr (not ReuseSmemC) {
            // Let producer load warp know smem buffers are consumed and empty
            cutlass::arch::fence_view_async_shared();
            load_pipeline.consumer_release(load_pipe_consumer_state);
            ++load_pipe_consumer_state;
          }
          ++load_wait_state;
        }

        int mma_m = epi_m;
        int mma_n = (epi_n * size<1>(EpilogueTile{})) / mma_tile_n;
        Tensor tRS_rAcc_frg_mn = tRS_rAcc_frg(_,mma_m,mma_n);

        // Vectorized fragment loop with visitor callback entry point
        int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
        int r2s_v = epi_n_in_mma * size(tRS_rCompute_frg);
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
          tRS_rCompute_frg(epi_v) = cst_callbacks.visit(tRS_rAcc_frg_mn(r2s_v + epi_v), epi_v, epi_m, epi_n);
        }
        // The latest we can delay the TMA store is right before the smem store of the next iteration
        // since the current TMA store needs to be committed before we can acquire the next smem buffer
        if constexpr (DelayTmaStore) {
          // Issue TMA stores for the previous subtile
          if (not is_first_iteration and subtile_idx == -1) {
            tma_store_fn(epi_m_prev, epi_n_prev);
          }
          epi_m_prev = epi_m;
          epi_n_prev = epi_n;
        }

        // Smem reduction callback entry point using current store buffer for workspace
        cst_callbacks.reduce(sD_epi(_,_,store_pipe_producer_state.index()),
                              synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);

        // Copy tile from register to regiser if needed
        if constexpr (IsUseR2R) {
          // retile source and destination for tiled_r2r
          Tensor tRR_rD_src = thread_r2r.retile_S(tRS_rCompute);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)
          Tensor tRR_rD_dst = thread_r2r.retile_D(tRS_rCompute);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)

          // Output register transformation before copying to shared memory.
          copy(tiled_r2r, tRR_rD_src, tRR_rD_dst);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tRS_rD_frg); ++i) {
          tRS_rD_frg(i) = cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(tRS_rCompute_frg(i));
        }

        // Copy tile from register to smem
        if constexpr (is_destination_supported) {
          copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));
        }

        // Post reduction, pre TMA store callback entry point
        constexpr bool issue_smem_store = true; // No smem store predication
        cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store);

        if constexpr (not DelayTmaStore) {
          // Issue TMA stores for this subtile
          tma_store_fn(epi_m, epi_n);
        }

        cst_callbacks.end_loop(epi_m, epi_n);

      } // for epi_m
    } // for epi_n

    if constexpr (DelayTmaStore) {
      // Issue TMA stores for the last subtile
      tma_store_fn(epi_m_prev, epi_n_prev);
    }

    // Post-loop fusion callback entry point
    cst_callbacks.end();

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state) {
    // wait for all TMA stores to complete
    store_pipeline.producer_tail(store_pipe_producer_state);
    // reset store counter
    issued_stores = 0;

    if constexpr (ReuseSmemC) {
      if (fusion_callbacks.is_producer_load_needed()) {
        // Issue releases on up to StagesD-1 previously issued TMA stores
        constexpr int release_stages = cute::min(StorePipeline::UnacquiredStages, get_load_pipe_increment(CtaTileMNK{}));
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < release_stages; ++stage) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
      }
    }

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
  int issued_stores = 0;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
