/***************************************************************************************************
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

#include <cmath>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#if defined(CUTLASS_ENABLE_SYCL)
#include "cutlass/util/reference/device/sycl_tensor_fill.h"
#else
#include "cutlass/util/reference/device/tensor_fill.h"
#endif
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

#include <helper.h>

#include "gemm_with_softmax_cute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

enum class Disposition {
  kPassed,
  kIncorrect,
  kNotVerified
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  int iterations;
  unsigned seed;
  float alpha;
  float beta;
  bool verification_enabled;
  float tolerance;

  Options():
    help(false),
    problem_size({16, 128, 64}),
    batch_count(16),
    iterations(20),
    seed(2022),
    alpha(1),
    beta(0),
    verification_enabled(true),
    tolerance(1e-5f)
  { }

  bool valid() {

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("batch_count", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("tolerance", tolerance);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "35_gemm_softmax example\n\n"
      << "  This example uses the CUTLASS Library to compute GEMM + Softmax for arbitrary problem sizes.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --batch_count=<int>         Batch number\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --seed=<int>                Random number seed (1*)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (0 to disable profiling).\n\n"
      << "  --verify=<bool>             If true, performs reference calculation.\n\n"
      << "  --tolerance <float>         Error tolerance\n"
    ;

    out << "\n\nExamples:\n\n"
      << "$ ./examples/35_gemm_softmax/35_gemm_softmax --m=1024 --n=512 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Returns true if the environment and Toolkit support this
  bool supported(bool verbose = true) const {

#if !defined(CUTLASS_ENABLE_SYCL)

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
      }
      return false;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      if (verbose) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
      }
      return false;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                  << std::endl;
      }
      return false;
    }
#endif
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  //
  // Type definitions
  //


  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = half_t;
  using ElementCompute = float;
  using ElementD = ElementC;
  using ElementSoftmax = ElementC;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using TileShape = Shape<_128, _128, _32>; // M, N, K 128-32

  using TiledMma = TiledMMA<
          MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
          Layout<Shape<_2,_2,_1>>,
          Tile<_32,_32,_16>>;

  using SmemLayoutAtomA = decltype(composition(Swizzle<2,3,3>{},
                Layout<Shape < _8,_32>,
                        Stride<_32, _1>>{}));
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementA>{},
                    Layout<Shape <_32,_4>,
                            Stride< _4,_1>>{},
                    Layout<Shape < _1,_8>>{}));

  using SmemLayoutAtomB = decltype(composition(Swizzle<2,3,3>{},
                Layout<Shape < _8,_32>,
                        Stride<_32, _1>>{}));
  using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementB>{},
                    Layout<Shape <_32,_4>,
                            Stride< _4,_1>>{},
                    Layout<Shape < _1,_8>>{}));

  using Stages = Int<3>;

  using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<Stages{}>;

  // ApplyShape impacts the final Softmax performance a lot.
  // Set ApplyShape::kColumn to be the next multiple of 32 number that is after
  // (gemm_N / alignment).
  // Set ApplyShape::kRow to max(1, 128 / ApplyShape::kColumn).
  using ApplyShape = cutlass::MatrixShape<1, 1024>;

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute
  >;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy,
          TileShape,
          ElementA,
          cutlass::detail::TagToStrideA_t<LayoutA>,
          ElementB,
          cutlass::detail::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
          GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using GemmSoftmax = cutlass::GemmSoftmax<
    CollectiveMainloop,
    EpilogueFunctorOp,
    ApplyShape
  >;

  using ElementNorm = typename GemmSoftmax::ElementNorm;
  using ElementSum = typename GemmSoftmax::ElementSum;
  using LayoutC = typename GemmSoftmax::LayoutC;
  using LayoutN = typename GemmSoftmax::LayoutN;
  using LayoutS = typename GemmSoftmax::LayoutS;
  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Data members
  //

  Options const &options;


  cutlass::HostTensor<ElementNorm, LayoutC>     reference_N;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<ElementD> block_Ref;
  cutlass::DeviceAllocation<ElementSoftmax> block_Softmax;
  cutlass::DeviceAllocation<ElementNorm> block_Norm;
  cutlass::DeviceAllocation<ElementSum> block_Sum;

  int block_num = (options.problem_size.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

  cutlass::gemm::GemmCoord problem = options.problem_size;

  int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
  int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
  int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

  // fixed rowmajor for norm and sum
  int64_t ldn = problem.m();
  int64_t lds = ldn;

  int64_t total_elements_A_per_batch = problem.m() * problem.k();
  int64_t total_elements_B_per_batch = problem.k() * problem.n();
  int64_t total_elements_C_per_batch = problem.m() * problem.n();
  int64_t total_elements_D_per_batch = problem.m() * problem.n();
  int64_t total_elements_partial_norm_per_batch = block_num * problem.m();

  int64_t total_elements_A = total_elements_A_per_batch * options.batch_count;
  int64_t total_elements_B = total_elements_B_per_batch * options.batch_count;
  int64_t total_elements_C = total_elements_C_per_batch * options.batch_count;
  int64_t total_elements_D = total_elements_D_per_batch * options.batch_count;
  int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * options.batch_count;

  //
  // Methods
  //

  Testbed(
    Options const &options_
  ):
    options(options_)
  {
    reference_N.reset({options.problem_size.m(), 1}, false);
  }

  /// Run
  Disposition run() {

    Disposition disposition = Disposition::kNotVerified;

    //
    // Initialize the workspace
    //

    initialize();

    //
    // Launch device kernel
    //
    cutlass::Status status = cutlass::Status::kSuccess;

    status = execute_device_kernel();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Device execution failed." << std::endl;
      return disposition;
    }

#if defined(CUTLASS_ENABLE_SYCL)
    syclcompat::wait();
#else
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return disposition;
    }
#endif

    //
    // Verify
    //

    if (options.verification_enabled) {

      bool passed = verify();

      if (passed) {
        disposition = Disposition::kPassed;
      }
      else {
        disposition = Disposition::kIncorrect;
      }
    }

    //
    // Profiling
    //
    if (options.iterations) {
      profile();
    }

    return disposition;
  }

  /// Random initialization
  void initialize() {

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_Softmax.reset(total_elements_D);
    block_Ref.reset(total_elements_D_per_batch);
    block_Norm.reset(total_elements_partial_norm);
    block_Sum.reset(total_elements_partial_norm);

    cutlass::reference::device::BlockFillRandomUniform(
            block_A.get(), total_elements_A, options.seed, ElementA(5), ElementA(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_B.get(), total_elements_B, options.seed + 1, ElementB(5), ElementB(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_C.get(), total_elements_C, options.seed + 2, ElementC(5), ElementC(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_D.get(), total_elements_D, options.seed + 3, ElementD(5), ElementD(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_Ref.get(), total_elements_D_per_batch, options.seed + 3, ElementD(5), ElementD(-5), 0);

    cutlass::reference::device::BlockFillRandomUniform(
            block_Softmax.get(), total_elements_D, options.seed + 3, ElementSoftmax(5), ElementSoftmax(-5), 0);

    cutlass::reference::host::TensorFill(
      reference_N.host_view(),
      ElementNorm()
    );

  }

  cutlass::Status execute_device_kernel() {

    cutlass::Status status = cutlass::Status::kSuccess;

    //
    // Setup arguments
    //

    // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID.
    // This information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    GemmSoftmax::Arguments args(
      options.problem_size,
      options.batch_count,
      {block_A.get(), lda},
      {block_B.get(), ldb},
      {block_C.get(), ldc},
      {block_D.get(), ldc},
      {
        ElementCompute(options.alpha),
        ElementCompute(options.beta)
      },
      {block_Norm.get(), ldn},
      {block_Sum.get(), lds},
      {block_Softmax.get(), ldc},
      hw_info,
      total_elements_A_per_batch,
      total_elements_B_per_batch,
      total_elements_C_per_batch,
      total_elements_D_per_batch,
      total_elements_partial_norm_per_batch,
      total_elements_partial_norm_per_batch,
      total_elements_D_per_batch
    );

    //
    // Launch
    //

    GemmSoftmax gemm_softmax;

    // Initialize
    status = gemm_softmax.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Run
    status = gemm_softmax();

    return status;
  }

  template<typename Element>
  bool verify_tensor(std::vector<Element> vector_Input, \
                       std::vector<Element> vector_Input_Ref) {

    auto size = int64_t((vector_Input.size() < vector_Input_Ref.size()) ? vector_Input.size() : vector_Input_Ref.size());
    float abs_tol = options.tolerance;
    float rel_tol = options.tolerance;
    
    for (int64_t i = 0; i < size; ++i) {
      float diff = (float)(vector_Input.at(i) - vector_Input_Ref.at(i));
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref.at(i));
      float relative_diff = abs_ref > abs_tol ? abs_diff / abs_ref : 0;
      if ( (isnan(abs_diff) || isinf(abs_diff)) ||  (abs_diff > rel_tol && relative_diff > rel_tol)) {
        printf("diff = %f, {%f, %f}.\n", abs_diff, (float)(vector_Input.at(i)), (float)(vector_Input_Ref.at(i)));
        return false;
      }

    }

    return true;
  }

  /// Verifies the reference matches
  bool verify() {

    LayoutA layout_A(lda);
    LayoutB layout_B(ldb);
    LayoutC layout_C(ldc);
    LayoutN Layout_N(ldn);
    LayoutS Layout_S(lds);

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};

    for (int batch_idx = 0; batch_idx < options.batch_count; batch_idx++) {

      cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + total_elements_A_per_batch * batch_idx, layout_A, extent_A);
      cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + total_elements_B_per_batch * batch_idx, layout_B, extent_B);
      cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + total_elements_C_per_batch * batch_idx, layout_C, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_C, extent_C);

      cutlass::reference::device::GemmComplex<
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementC, LayoutC, 
          ElementCompute, ElementCompute
      >(
        problem,
        options.alpha, 
        view_A,
        cutlass::ComplexTransform::kNone,
        view_B,
        cutlass::ComplexTransform::kNone,
        options.beta, 
        view_C, 
        view_Ref_device, 
        ElementCompute(0)
      );

#if defined(CUTLASS_ENABLE_SYCL)
      syclcompat::wait();
#endif

      // Copy reference results to host memory for verification
      std::vector<ElementD> matrix_D_Ref(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_D_Ref.data(), block_Ref.get(), matrix_D_Ref.size());
      cutlass::TensorView<ElementD, LayoutC> view_Ref(matrix_D_Ref.data(), layout_C, extent_C);

      std::vector<ElementSoftmax> matrix_Softmax_Ref(layout_C.capacity(extent_C));
      cutlass::TensorView<ElementSoftmax, LayoutC> view_Softmax_Ref(matrix_Softmax_Ref.data(), layout_C, extent_C);

      // Copy computed results to host memory
      std::vector<ElementD> matrix_D(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + total_elements_D_per_batch * batch_idx, matrix_D.size());

      std::vector<ElementD> matrix_Softmax(layout_C.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_Softmax.data(), block_Softmax.get() + total_elements_D_per_batch * batch_idx, matrix_Softmax.size());

      // std::vector<float> norm(total_elements_partial_norm);
      // cutlass::device_memory::copy_to_host(norm.data(), block_Norm.get(), norm.size());
      // std::vector<float> sum(total_elements_partial_norm);
      // cutlass::device_memory::copy_to_host(sum.data(), block_Sum.get(), sum.size());
      //
      // for (int i = 0; i < norm.size(); i++) {
      //   printf("norm[%d] = %f\n", i, norm[i]);
      // }
      //
      // for (int i = 0; i < sum.size(); i++) {
      //   printf("sum[%d] = %f\n", i, sum[i]);
      // }

      // Compute the norm
      for (int m = 0; m < options.problem_size.m(); ++m) {
        reference_N.at({m, 0}) = view_Ref.ref().at({m, 0});
        for (int n = 1; n < options.problem_size.n(); ++n) {
          reference_N.at({m, 0}) = std::max(reference_N.at({m, 0}), ElementNorm(view_Ref.ref().at({m, n})));
        }
      }

      // Compute softmax
      for (int m = 0; m < options.problem_size.m(); ++m) {

        float sum = float();

        for (int n = 0; n < options.problem_size.n(); ++n) {
          sum += std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) );
        }

        float inv_sum = float(1.0f / sum);

        for (int n = 0; n < options.problem_size.n(); ++n) {

          view_Softmax_Ref.ref().at({m, n}) = ElementSoftmax(
            std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) ) * inv_sum
          );
        }
      }

      // Verification checks - set any of these to 'true' to override the verification checks.
      bool verified_D = false;
      bool verified_Softmax = false;

      // Verify softmax output
      if (!verified_D) {
        verified_D = verify_tensor<ElementC>(matrix_D, matrix_D_Ref);
      }

      if (!verified_Softmax) {
        verified_Softmax = verify_tensor<ElementSoftmax>(matrix_Softmax, matrix_Softmax_Ref);
      }

      if (!verified_D || !verified_Softmax) {

        std::cerr << "Verification check failed for tensor Softmax at batch " << batch_idx << "\n";

        // Summarize which checks failed
        if (!verified_D) {
          std::cerr << "Verification of D tensor failed\n";
        }

        if (!verified_Softmax) {
          std::cerr << "Verification of Softmax tensor failed\n";
        }

        return false;
      }

    }

    return true;
  }

  /// Profiles
  bool profile() {

    //
    // Profile
    //

    cutlass::Status status = cutlass::Status::kSuccess;
    GpuTimer timer;
    int const kIterations = options.iterations;

    timer.start();
    for (int iter = 0; iter < kIterations; ++iter) {

      status = execute_device_kernel();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Device execution failed." << std::endl;
        return false;
      }
    }
    timer.stop();

    float elapsed_ms = timer.elapsed_millis();

    int64_t flops = int64_t(options.problem_size.m()) * options.problem_size.n() * options.problem_size.k() * 2;
    int64_t bytes = (sizeof(ElementD) * 2 + sizeof(ElementSoftmax)) * options.problem_size.m() * options.problem_size.n();

    double gflops_per_second = double(flops) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1.0e9);
    double gbytes_per_second = double(bytes) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1 << 30);

    double elapsed_ms_per_iter = double(elapsed_ms) / kIterations;

    std::cout << "         Problem: "
              << options.problem_size.m() << "-by-" << options.problem_size.n() << "-by-" << options.problem_size.k()
              << ", batch size: " << options.batch_count
              << std::endl;

    std::cout << "         Runtime: " << elapsed_ms_per_iter << " ms\n" << std::endl;

    std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
    std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  // Options parsing
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Run
  Testbed testbed(options);

  Disposition disposition = testbed.run();

  std::cout << std::endl;

  switch (disposition) {
    case Disposition::kPassed:
      std::cout << "Passed" << std::endl;
      break;
    case Disposition::kIncorrect:
      std::cout << "Incorrect" << std::endl;
      break;
    case Disposition::kNotVerified:
      std::cout << "Not verified" << std::endl;
      break;
  }

  return (disposition == Disposition::kPassed ? 0 : -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
