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

/**

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"

#include "cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/reduction/kernel/reduce_softmax_final.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

#include "gemm_with_epilogue_visitor.h"
#include "epilogue_with_softmax_cute.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernel computes partial reduction
//
//
// 2. Sum[m, n'] = sum_n(exp(D[m, n] - N[m, 0]))
//
template <
  typename ElementD_,
  typename ElementNorm_,
  typename ElementSum_,
  typename ElementSoft_,
  typename ElementSoftmaxCompute_,
  int Alignment,
  typename ApplyShape_ = MatrixShape<1, 1024>
>
class ApplySoftmax {
public:

  using ElementD = ElementD_;
  using ElementNorm = ElementNorm_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoft_;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;

  static int const kAlignment = Alignment;
  using ApplyShape = ApplyShape_;

  using Layout = cutlass::layout::RowMajor;

  using TensorRefD = TensorRef<ElementD, Layout>;
  using TensorRefN = TensorRef<ElementNorm, Layout>;
  using TensorRefSum = TensorRef<ElementSum, Layout>;
  using TensorRefSoft = TensorRef<ElementSoft, Layout>;

  using FragmentSoftmax = Array<ElementSoftmaxCompute, kAlignment>;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord     extent;             ///< Extent of D and Softmax matrices
    int             batch_count;        ///< Batch count
    TensorRefD      ref_D;              ///< D matrix computed by GEMM+Max (input)
    TensorRefN      ref_N;              ///< Norm tensor (input)
    TensorRefSum    ref_S;              ///< Sum  tensor (input)
    TensorRefSoft   ref_Soft;           ///< Softmax tensor (output)
    int64_t         batch_stride_D;     ///< Batch stride for D tensor
    int64_t         batch_stride_N;     ///< Batch stride for N tensor
    int64_t         batch_stride_S;     ///< Batch stride for S tensor
    int64_t         batch_stride_Soft;  ///< Batch stride for softmax tensor

    //
    // Methods
    //
    Arguments():
      batch_count(1),
      batch_stride_D(0),
      batch_stride_N(0),
      batch_stride_S(0),
      batch_stride_Soft(0)
    { }

    Arguments(
      MatrixCoord     extent_,             ///< Extent of D and Softmax matrices
      int             batch_count_,        ///< Batch count
      TensorRefD      ref_D_,              ///< D matrix computed by GEMM+PartialReduce
      TensorRefN      ref_N_,              ///< Output parameter for N
      TensorRefSum    ref_S_,              ///< Output parameter for N
      TensorRefSoft   ref_Soft_,           ///< Softmax
      int64_t         batch_stride_D_ = 0,
      int64_t         batch_stride_N_ = 0,
      int64_t         batch_stride_S_ = 0,
      int64_t         batch_stride_Soft_ = 0
    ):
      extent(extent_),
      batch_count(batch_count_),
      ref_D(ref_D_),
      ref_N(ref_N_),
      ref_S(ref_S_),
      ref_Soft(ref_Soft_),
      batch_stride_D(batch_stride_D_),
      batch_stride_N(batch_stride_N_),
      batch_stride_S(batch_stride_S_),
      batch_stride_Soft(batch_stride_Soft_)
    {

    }
  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

  //
  // SharedStorage
  //

  struct SharedStorage {

  };

private:

public:

  CUTLASS_DEVICE
  ApplySoftmax() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    apply(params, shared_storage);
  }

private:


  /// Compute Softmax
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    int block_batch = BlockIdxZ();
    int block_m = BlockIdxX() * ApplyShape::kRow;
    int block_n = 0;

    int thread_m = ThreadIdxY();
    int thread_n = ThreadIdxX() * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    int batch_offset_norm = block_batch * params.args.batch_stride_N;
    int batch_offset_sum = block_batch * params.args.batch_stride_S;

    // Kill off thread if it is outside the row boundary
    if (params.args.extent.row() <= idx_m) {
      return;
    }

    //
    // Setup pointers to load D again
    //

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;
    using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;
    using FragmentSoft = Array<ElementSoft, kAlignment>;
    using ConvertSoftCompute = cutlass::NumericArrayConverter<ElementSoftmaxCompute, ElementD, kAlignment>;
    using ConvertSoftOutput = cutlass::NumericArrayConverter<ElementSoft, ElementSoftmaxCompute, kAlignment>;

    using Mul = cutlass::multiplies<FragmentSoftmax>;
    using Minus = cutlass::minus<FragmentSoftmax>;
    using Exp   = cutlass::fast_exp_op<FragmentSoftmax>;

    ConvertSoftCompute   convert_soft_compute;
    ConvertSoftOutput  convert_soft_output;

    Minus     minus;
    Mul       mul;
    Exp       exponential;

    using ConvertSum = cutlass::NumericConverter<ElementSoftmaxCompute, ElementSum>;
    using ConvertNorm = cutlass::NumericConverter<ElementSoftmaxCompute, ElementNorm>;

    ConvertSum   convert_sum;
    ConvertNorm  convert_norm;

    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
      params.args.ref_D.data() +
      params.args.batch_stride_D * block_batch +
      params.args.ref_D.layout()({idx_m, idx_n}));

    AccessTypeSoft *access_soft = reinterpret_cast<AccessTypeSoft *>(
      params.args.ref_Soft.data() +
      params.args.batch_stride_Soft * block_batch +
      params.args.ref_Soft.layout()({idx_m, idx_n}));

    ElementSum inv_sum = (params.args.ref_S.data())[idx_m + batch_offset_sum];
    ElementNorm norm = (params.args.ref_N.data())[idx_m + batch_offset_norm];

    //
    // Loop
    //
    CUTLASS_PRAGMA_UNROLL
    for (
      int idx = 0;
      idx < params.args.extent.column();
      idx += ApplyShape::kColumn * kAlignment) {

      if (idx_n < params.args.extent.column()) {
        AccessTypeD fetch;
        arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d, true);

        FragmentSoftmax result = mul(exponential(minus(convert_soft_compute(fetch), convert_norm(norm))),  convert_sum(inv_sum));
        FragmentSoft soft  = convert_soft_output(result);

        arch::global_store<FragmentSoft, sizeof(FragmentSoft)>(soft, access_soft, true);
      }

      access_d += ApplyShape::kColumn;
      access_soft += ApplyShape::kColumn;
      idx_n += ApplyShape::kColumn * kAlignment;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename CollectiveMainloop_,
  typename EpilogueFunctorOp_,
  typename ApplyShape_ = MatrixShape<1, 1024>,
  int AlignmentA_ = 128 / cutlass::sizeof_bits<typename CollectiveMainloop_::ElementA>::value,
  int AlignmentB_ = 128 / cutlass::sizeof_bits<typename CollectiveMainloop_::ElementB>::value,
  int AlignmentSoftmax_ = 128 / cutlass::sizeof_bits<typename EpilogueFunctorOp_::ElementC>::value,
  typename ElementNorm_ = float,
  typename ElementSum_ = float,
  typename ElementSoftmax_ = typename EpilogueFunctorOp_::ElementC
>
class GemmSoftmax {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using CollectiveMainloop = CollectiveMainloop_;
  using EpilogueFunctorOp = EpilogueFunctorOp_;

  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementC = typename EpilogueFunctorOp::ElementC;
  using ElementCompute = typename EpilogueFunctorOp::ElementCompute;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoftmax_;
  using ElementSoftmaxCompute = float;

  using ElementNorm = ElementNorm_;

  using ApplyShape = ApplyShape_;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutS = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementNorm, LayoutN>;
  using TensorRefSum = TensorRef<ElementSum, LayoutS>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;
  static int const AlignmentSoftmax = AlignmentSoftmax_;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using PreSwizzleLayout = Layout< Shape<_128,_128>, Stride<_1, _128>>;

  using SmemLayout = ComposedLayout<
                     Swizzle<3,4,3>,
                     smem_ptr_flag_bits<sizeof_bits<float>::value>,
                     PreSwizzleLayout>;

  // 128 threads loading 16 elements each (to get vectorized global stores)
  using TileShapeS2R = Shape<_32,_8>;

  // Tiled copy from Smem to Registers
  // Note : CuTe will vectorize this copy if the tiling + swizzling above were right
  using TiledCopyS2R = TiledCopy<
                         Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half>,
                         Layout< Shape<_32,_8>,
                                 Stride<_1,_32>>,
                         TileShapeS2R>;

  using CollectiveEpilogue = cutlass::epilogue::collective::EpilogueWithSoftmax<
        cutlass::detail::TagToStrideC_t<LayoutC>,
        cutlass::detail::TagToStrideC_t<LayoutC>,
        EpilogueFunctorOp,
        SmemLayout,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half>,
        TiledCopyS2R,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, half>>;

  // basic GEMM kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>,
          CollectiveMainloop,
          CollectiveEpilogue
  >;

  // GEMM
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Softmax kernel
  using SoftmaxApplyKernel = kernel::ApplySoftmax<
    ElementC,
    ElementNorm,
    ElementSum,
    ElementSoft,
    ElementSoftmaxCompute,
    AlignmentSoftmax,
    ApplyShape
  >;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using ApplyFinalReductionKernel = cutlass::reduction::kernel::ApplySoftmaxFinalReduction<
    ElementNorm,
    ElementSum,
    ElementSoftmaxCompute,
    ThreadblockShape
  >;

public:

  /// Arguments class
  struct Arguments {
    typename Gemm::Arguments         gemm;
    typename SoftmaxApplyKernel::Arguments softmax;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      int32_t    batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      typename EpilogueFunctorOp::Params linear_scaling,
      TensorRefN ref_N_,
      TensorRefSum ref_S_,
      TensorRefSoft ref_Softmax_,
      cutlass::KernelHardwareInfo hw_info,
      int64_t batch_stride_A_ = 0,
      int64_t batch_stride_B_ = 0,
      int64_t batch_stride_C_ = 0,
      int64_t batch_stride_D_ = 0,
      int64_t batch_stride_Max_ = 0,
      int64_t batch_stride_Sum_ = 0,
      int64_t batch_stride_Softmax_ = 0
    ):
      gemm({
        gemm::GemmUniversalMode::kGemm,
        typename Gemm::GemmKernel::ProblemShape(problem_size.m(), problem_size.n(), problem_size.k(), batch_count_),
        {
          ref_A_.data(),
          cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(problem_size.m(), problem_size.k(), batch_count_)),
          ref_B_.data(),
          cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(problem_size.n(), problem_size.k(), batch_count_))
        },
        {
          linear_scaling,
          ref_C_.data(),
          cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, make_shape(problem_size.m(), problem_size.n(), batch_count_)),
          ref_D_.data(),
          cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, make_shape(problem_size.m(), problem_size.n(), batch_count_)),
          ref_S_.data(),
          ref_N_.data()
        },
        hw_info
      }),
      reduction(
        problem_size,
        ref_N_.data(),
        ref_S_.data(),
        batch_stride_Max_,
        batch_stride_Sum_
      ), 
      softmax(
        MatrixCoord(problem_size.m(), problem_size.n()),
        batch_count_,
        ref_D_,
        ref_N_,
        ref_S_,
        ref_Softmax_,
        batch_stride_D_,
        batch_stride_Max_,
        batch_stride_Sum_,
        batch_stride_Softmax_
      ),
      extend(problem_size)
    {
    }
  };

  struct Params {
    typename Gemm::Params                      gemm;
    typename SoftmaxApplyKernel::Params        softmax;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord                                extend;
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Gemm gemm{};
  Params params_;

public:

  /// Ctor
  GemmSoftmax() {

  }

  static Status
  can_implement(Arguments const& args) {
    if (Gemm::can_implement(args)) {
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Gemm::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    // Initialize Gemm
    Status status = gemm.initialize(args.gemm, workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = Params{
      gemm.params(),
      typename SoftmaxApplyKernel::Params(args.softmax),
      typename ApplyFinalReductionKernel::Params(args.reduction),
      MatrixCoord(args.extend)
      };
    return Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the GEMM + max kernel
    //

    cudaError_t result;

    gemm.run();

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }


    //
    // Launch the ApplyFinalReductionKernel
    //

    int thread_per_block = 128;
    int block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_grid(block_per_row, 1, params_.softmax.args.batch_count);
    dim3 final_reduction_block(thread_per_block);

#if defined(CUTLASS_ENABLE_SYCL)
    const auto sycl_final_reduction_block = syclcompat::dim3(final_reduction_block.x, final_reduction_block.y, final_reduction_block.z);
    const auto sycl_final_reduction_grid = syclcompat::dim3(final_reduction_grid.x, final_reduction_grid.y, final_reduction_grid.z);

    using namespace syclcompat::experimental;

    auto final_reduction_event = launch<Kernel<ApplyFinalReductionKernel>>(launch_policy{
      sycl_final_reduction_grid, sycl_final_reduction_block, local_mem_size{sizeof(typename ApplyFinalReductionKernel::SharedStorage)}},
      params_.reduction);
    EventManager::getInstance().addEvent(final_reduction_event);
#else
    Kernel<ApplyFinalReductionKernel><<<
      final_reduction_grid, final_reduction_block, sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream
    >>>(params_.reduction);
#endif

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(SoftmaxApplyKernel::ApplyShape::kColumn, SoftmaxApplyKernel::ApplyShape::kRow);

    int threadblock_rows = SoftmaxApplyKernel::ApplyShape::kRow;
    int threadblock_columns = SoftmaxApplyKernel::ApplyShape::kColumn * SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
      (params_.softmax.args.extent.row() + threadblock_rows - 1) / threadblock_rows,
      (params_.softmax.args.extent.column() + threadblock_columns - 1) / threadblock_columns,
      params_.softmax.args.batch_count);

#if defined(CUTLASS_ENABLE_SYCL)
    const auto sycl_apply_block = syclcompat::dim3(apply_block.x, apply_block.y, apply_block.z);
    const auto sycl_apply_grid = syclcompat::dim3(apply_grid.x, apply_grid.y, apply_grid.z);

    using namespace syclcompat::experimental;

    auto apply_event = launch<Kernel<SoftmaxApplyKernel>>(launch_policy{
      sycl_apply_grid, sycl_apply_block, local_mem_size{sizeof(typename SoftmaxApplyKernel::SharedStorage)}},
      params_.softmax);
    EventManager::getInstance().addEvent(apply_event);
#else
    Kernel<SoftmaxApplyKernel><<<
      apply_grid, apply_block, sizeof(typename SoftmaxApplyKernel::SharedStorage), stream
    >>>(params_.softmax);
#endif

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
