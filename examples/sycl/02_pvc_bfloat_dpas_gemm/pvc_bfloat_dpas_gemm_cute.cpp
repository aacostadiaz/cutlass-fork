/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

bool identityData = false;
bool fixedData = true;
bool validate = true;
float threshold = 0.01f;

template <typename T>
static void fill_matrix(std::vector<T> &M, size_t numRows, size_t numCols)
{
    if (identityData)
    {
        std::generate(std::begin(M), std::end(M), [&]
                      { return static_cast<T>(1); });
    }
    else if (fixedData)
    {
        for (int r = 0; r < numRows; r++)
        {
            for (int c = 0; c < numCols; c++)
            {
                M[r * numCols + c] = static_cast<T>(r + c);
            }
        }
    }
    else
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(1.0, 2.0);
        std::generate(std::begin(M), std::end(M), [&]
                      { return static_cast<T>(dist(rng)); });
    }
}

template <typename T>
static void vnni_matrix(
    std::vector<T> &dst, const std::vector<T> &src,
    size_t numRows, size_t numCols, size_t factor)
{
    for (size_t r = 0; r < numRows / factor; r++) {
        for (size_t c = 0; c < numCols; c++) {
            for (size_t k = 0; k < factor; k++) {
                dst[r * numCols * factor + c * factor + k] =
                    src[(r * factor + k) * numCols + c];
            }
        }
    }
}

template <typename DstT, typename SrcT>
static void compute_reference(
    std::vector<DstT>& C,
    const std::vector<SrcT>& A, const std::vector<SrcT>& B,
    size_t M, size_t N, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            DstT sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum = std::fma(static_cast<DstT>(A[m * K + k]),
                               static_cast<DstT>(B[k * N + n]), sum);
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
bool check_results(
    size_t M,
    size_t N,
    const std::vector<T>& C,
    const std::vector<T>& C_ref)
{
    float err = 0.f;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto index = m * N + n;
            auto localErr = std::fabs(C[index] - C_ref[index]) /
                            std::max(std::fabs(C[index]),
                                    std::fabs(C_ref[index]));
            err = std::max(localErr, err);
            if (localErr >= threshold) {
                std::cerr << "Error at m = " << m << ", n = " << n
                          << ": (local error " << localErr << "): Wanted "
                          << C_ref[index] << ", got " << C[index] << std::endl;
                return false;
            }
        }
    }
  return true;
}

using namespace cute;

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = bfloat16_t;                        // <- data type of elements in input matrix A
using ElementInputB = bfloat16_t;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

using TileShape = Shape<_32, _32, _16>;

using TiledMma = TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
        Layout<Shape<_8,_16,_1>>>;

//// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        // memory access. For a byte, it's 16
        // elements. This becomes the vector width of
        // math instructions in the epilogue too
        ElementAccumulator,                                // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using DispatchPolicy = cutlass::gemm::MainloopIntelPVCUnpredicated;

template <typename Gemm_Op>
void
run(Gemm_Op gemm_op)
{
  gemm_op();
}

void test_gemm(int m, int n, int k)
{
  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = bfloat16_t;
  using TB = bfloat16_t;
  using TC = float;
  using TI = float;

  std::vector<TA> h_A(m*k);
  std::vector<TB> h_B(n*k);
  std::vector<TB> h_B_vnni(n*k);
  std::vector<TC> h_C(m*n, static_cast<TC>(0));
  std::vector<TC> h_D(m*n, static_cast<TC>(0));

  fill_matrix(h_A, m, k);
  fill_matrix(h_B, k, n);

  vnni_matrix(h_B_vnni, h_B, k, n, 2);

  auto d_A = syclcompat::malloc<TA>(m*k);
  auto d_B = syclcompat::malloc<TB>(n*k);
  auto d_C = syclcompat::malloc<TC>(m*n);

  syclcompat::memcpy<TA>(d_A, h_A.data(), h_A.size());
  syclcompat::memcpy<TB>(d_B, h_B_vnni.data(), h_B_vnni.size());
  syclcompat::memcpy<TC>(d_C, h_C.data(), h_C.size());

  TI alpha = 1.0;
  TI beta  = 0.0;

  double tflops = (2.0*m*n*k) * 1e-12;

  const int timing_iterations = 10;
  GPU_Clock timer;

  //
  // CuTe
  //

  using namespace cute;

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, k, Int<1>{});
  auto dB = make_stride(Int<1>{}, n, Int<1>{});
  auto dC = make_stride(Int<1>{}, n, Int<1>{});

  using GmemTiledCopyA = XE_2D_LOAD;
  using GmemTiledCopyB = XE_2D_LOAD;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
          decltype(dC),
          decltype(dC),
          EpilogueOp,
          cutlass::gemm::EpilogueDefault>;

// Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy,
          TileShape,
          ElementInputA,
          decltype(dA),
          ElementInputB,
          decltype(dB),
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType cute_problem_size = ProblemShapeType{m, n, k, 1};

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          cute_problem_size,  // <- problem size of matrix multiplication
          {  d_A, dA, d_B, dB },
          {
                  { alpha, beta },
                  d_C, dC, d_C, dC
          }
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace = syclcompat::malloc<TA>(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  gemm_op.can_implement(arguments);
  CUTE_CHECK_LAST();

  // Initialize CUTLASS kernel with arguments and workspace pointer
  gemm_op.initialize(arguments, workspace);
  CUTE_CHECK_LAST();

  // Run once (and check)
  run(gemm_op);
  CUTE_CHECK_LAST();
  syclcompat::wait();
  syclcompat::memcpy<TC>(h_C.data(), d_C, h_C.size());
  syclcompat::wait();
  printf("Computing Reference\n");
  compute_reference(h_D, h_A, h_B, m, n, k);

  if(!check_results(m, n, h_D, h_C)) {
    printf("Incorrect output\n");
    syclcompat::free(reinterpret_cast<void*>(d_A));
    syclcompat::free(reinterpret_cast<void*>(d_B));
    syclcompat::free(reinterpret_cast<void*>(d_C));
    return;
  }
  printf("Correct output\n");
  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    run(gemm_op);
    syclcompat::wait();
  }
  CUTE_CHECK_LAST();
  double cute_time = timer.seconds() / timing_iterations;
  printf("CUTLASS_GEMM:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

  syclcompat::wait();
  syclcompat::free(reinterpret_cast<void*>(d_A));
  syclcompat::free(reinterpret_cast<void*>(d_B));
  syclcompat::free(reinterpret_cast<void*>(d_C));
}

int main(int argc, char** argv)
{
  int m = 64;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 64;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 64;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  test_gemm(m, n, k);

  return 0;
}
