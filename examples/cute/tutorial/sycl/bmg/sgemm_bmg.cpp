/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class TiledCopyA,
          class TB, class BStride, class TiledCopyB,
          class TC, class CStride, class TiledCopyC, class TiledMma,
          class Alpha, class Beta>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, int stages,
            TA const* A, AStride dA, TiledCopyA,
            TB const* B, BStride dB, TiledCopyB,
            TC      * C, CStride dC, TiledCopyC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  //
  // Full and Tiled Tensors
  //

  auto A_shape = select<0,2,3>(shape_MNK);
  auto B_shape = select<1,2,3>(shape_MNK);
  auto C_shape = select<0,1,3>(shape_MNK);

  // Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
  auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));
  auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));

  auto copy_a = TiledCopyA{mA};
  auto copy_b = TiledCopyB{mB};
  auto copy_c = TiledCopyC{mC};

  Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
  Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,l)
  Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(syclcompat::work_group_id::x(), syclcompat::work_group_id::y(), 0);  // (m,n,k)
  Tensor gA = local_tile(mA_coord, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB_coord, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC_coord, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  //
  // Define A/B partitioning and C accumulators
  //

  TiledMma tiled_mma;
  constexpr int sg_size = 16;
  auto sg = syclcompat::get_nd_item<1>().get_sub_group();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * sg_size;
  auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

  // Partition global counting tensors for MMA
  Tensor tCgA = thr_mma.partition_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = make_tensor<TA>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
  Tensor tCrB = make_tensor<TB>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

  ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
  ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());

  // Retile registers for copies
  Tensor tArA = thr_copy_a.retile_D(tCrA);
  Tensor tBrB = thr_copy_b.retile_D(tCrB);

  // Retile global counting tensors for copies
  Tensor tAgA = thr_copy_a.retile_S(tCgA);
  Tensor tBgB = thr_copy_b.retile_S(tCgB);

  //
  // PREFETCH
  //

  constexpr int Num_SGs = size(tiled_mma);
//  auto prefetch_a = cute::prefetch_selector<select<0,2>(cta_tiler), Num_SGs>(copy_a);
//  auto prefetch_b = cute::prefetch_selector<select<1,2>(cta_tiler), Num_SGs>(copy_b);
//  auto thr_prefetch_A = prefetch_a.get_slice(syclcompat::local_id::x());
//  auto thr_prefetch_B = prefetch_b.get_slice(syclcompat::local_id::x());

  // Partition global tile for prefetch
//  auto pAgA = thr_prefetch_A.partition_S(gA);
//  auto pBgB = thr_prefetch_B.partition_S(gB);

  int prefetch_k = 0;

  CUTE_UNROLL
  for (; prefetch_k < stages; prefetch_k++) {
//    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  // Clear the accumulators
  Tensor tCrC = partition_fragment_C(tiled_mma, take<0,2>(cta_tiler));
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  constexpr int barrier_scope = 2;
  int k_tile_count = ceil_div(get<2>(shape_MNK), get<2>(cta_tiler));

  CUTLASS_PRAGMA_UNROLL
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);
    // Copy gmem to rmem for the first k_tile
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    if (prefetch_k < k_tile_count) {
//      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }

#endif

  //
  // Epilogue
  //
  copy(copy_c, tCrC, tCgC(_,_,_,0));

}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto L = int(1);
  auto prob_shape = make_shape(M, N, K, L);                     // (M, N, K, L)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<256>{};
  auto bN = Int<256>{};
  auto bK = Int< 32>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<2>{};  // Pipeline

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, decltype(dA)>, TA>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                    Layout<Shape<_16,_1>>{});              // Val layout  32x2
  TiledCopy copyB = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_V, decltype(dB)>, TB>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_32,_2>>{});              // Val layout  16x1
  TiledCopy copyC = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, decltype(dC)>, TC>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_8,_1>>{});              // Val layout  8x1

  TiledMMA mmaC = TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<decltype(cta_tiler)>,
                                 Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = syclcompat::dim3(size(mmaC));
  auto dimGrid  = syclcompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  auto event = syclcompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(copyA),
                  TB, decltype(dB), decltype(copyB),
                  TC, decltype(dC), decltype(copyC), decltype(mmaC),
                  Alpha, Beta>>(dimGrid, dimBlock, prob_shape, cta_tiler, bP,
                    A, dA, copyA,
                    B, dB, copyB,
                    C, dC, copyC, mmaC,
                    alpha, beta);
  EventManager::getInstance().addEvent(event);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto L = int(1);
  auto prob_shape = make_shape(M, N, K, L);                     // (M, N, K, L)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{}, Int<0>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{}, Int<0>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC, Int<0>{});                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<256>{};
  auto bN = Int<256>{};
  auto bK = Int< 32>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<2>{};  // Pipeline

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, decltype(dA)>, TA>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                                    Layout<Shape<_32,_2>>{});              // Val layout  32x2
  TiledCopy copyB = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, decltype(dB)>, TB>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_16,_1>>{});              // Val layout  16x1
  TiledCopy copyC = make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, decltype(dC)>, TC>{},
                                    Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                                    Layout<Shape<_8,_1>>{});              // Val layout  8x1

  TiledMMA mmaC = TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<decltype(cta_tiler)>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};  // 256x128x16 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = syclcompat::dim3(size(mmaC));
  auto dimGrid  = syclcompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  auto event = syclcompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(copyA),
                  TB, decltype(dB), decltype(copyB),
                  TC, decltype(dC), decltype(copyC), decltype(mmaC),
                  Alpha, Beta>>(dimGrid, dimBlock, prob_shape, cta_tiler, bP,
                    A, dA, copyA,
                    B, dB, copyB,
                    C, dC, copyC, mmaC,
                    alpha, beta);
  EventManager::getInstance().addEvent(event);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  }
  assert(false && "Not implemented");
}


int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'N';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'T';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  std::vector<TA> h_A(m*k);
  std::vector<TB> h_B(n*k);
  std::vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  auto d_A = syclcompat::malloc<TA>(m*k);
  auto d_B = syclcompat::malloc<TB>(k*n);
  auto d_C = syclcompat::malloc<TC>(m*n);

  syclcompat::memcpy<TA>(d_A, h_A.data(), m*k);
  syclcompat::memcpy<TB>(d_B, h_B.data(), k*n);
  syclcompat::memcpy<TC>(d_C, h_C.data(), m*n);

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  gemm(transA, transB, m, n, k,
       alpha,
       d_A, ldA,
       d_B, ldB,
       beta,
       d_C, ldC);
  syclcompat::wait_and_throw();

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A, ldA,
         d_B, ldB,
         beta,
         d_C, ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}
