/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <cute/config.hpp>           // CUTE_HOST_DEVICE
#include <cute/layout.hpp>           // cute::Layout
#include <cute/layout_composed.hpp>  // cute::ComposedLayout
#include <cute/swizzle.hpp>          // cute::Swizzle, cute::get_swizzle primary template

/* Specialized functionality for a ComposedLayout of the form
 *   InvolutionFn o Offset o LayoutB
 * where the InvolutionFn is a Swizzle<B,M,S> and is not linear (hence the need for the Offset).
 *
 * Because these are specializations for core functions of ComposedLayout, these Swizzle Layouts
 * provide similar functionality to Layout including tiling, partitioning,
 * coordinate-to-index mapping and layout manipulations, but are not considered "normal" layouts.
 * For example, these provide shape() and size() functions, but do not provide stride() functions.
 *
 * Furthermore, each of these specializations uses Swizzle<>-specific knowledge in its implementation and
 * attempts to decay itself to a normal-layout with dynamic or static strides when certain slicing conditions
 * are met. This is possible by determining the subdomain of the Swizzle<> function that is identity and
 * testing if LayoutB's codomain is contained within it. In general, MizedBits is used as the Offset to track
 * statically-vs-dynamically known bits in the Offset to improve the decay to static or dynamic normal layouts.
 */

namespace cute
{

//
// Helper Function
//
template <int B, int M, int S, class Offset, class LayoutB>
struct get_swizzle<ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB>> { using type = Swizzle<B,M,S>; };

//
// Constructors
//

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Swizzle<B,M,S> const& sxor)
{
  return composition(sxor, Layout<Int<M+B+abs(S)>,Int<1>>{});
}

namespace detail {

template <int B, int M, int S, class OldShape, class OldStride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto
transfer_swizzle(Layout<OldShape,OldStride> const& old_layout,
                 Layout<NewShape,NewStride> const& new_layout)
{
  // Our goal is to determine a new swizzle for the strides in new_layout for consistent vectorizations

  // This is accomplished by identifying
  //  S o L  :=:  S? o L*
  // We identify the "active" portion of S by computing (P o L)(c*) where P is a projection generated by S
  // Then that active identifier is transformed through the layouts:
  //  L*(L[(P o L)(c*)])
  // which is a new swizzle identifier for S?, the new swizzle

  // Projections of the swizzle layout for composition, P
  auto swizzle_only_zy = make_layout(make_shape (Int<(1 << M)>{}, Int<(1 << B)>{}, Int<(1 << (abs(S)-B))>{}, Int<(1 <<  B        )>{}, Int<1>{}),
                                     make_stride(       Int<0>{}, Int<(1 << M)>{},                 Int<0>{}, Int<(1 << (M+abs(S)))>{}, Int<0>{}));

  // Compose with the tile to get the swizzle projection, P o L  [The Z and Y contributing portions of L]
  auto layout_only_zy       = composition(swizzle_only_zy, old_layout);
  // Transform the end coordinate to get the active bits of the swizzle, (P o L)(c*)
  auto swizzle_active_bits  = layout_only_zy(size(layout_only_zy)-Int<1>{});

  // Get the Z bit and the Y bits -- keep only those that are active in Z *and* Y
  auto zzz_msk = typename Swizzle<B,M,S>::zzz_msk{};
  auto yyy_msk = typename Swizzle<B,M,S>::yyy_msk{};
  auto msk_sft = typename Swizzle<B,M,S>::msk_sft{};
  auto active_Z = swizzle_active_bits & shiftr(swizzle_active_bits,  msk_sft) & zzz_msk;
  auto active_Y = swizzle_active_bits & shiftr(swizzle_active_bits, -msk_sft) & yyy_msk;

  // Pass the identifiers through the old layout and new layout to make a new swizzle identifier, L*(L[(P o L)(c*)])
  auto new_active_Z = new_layout(old_layout.get_1d_coord(active_Z));
  auto new_active_Y = new_layout(old_layout.get_1d_coord(active_Y));

  // Use this new swizzle identifier to construct the new swizzle for new_layout
  //   (this also makes sure it's a "valid" swizzle that Swizzle can represent)
  return composition(make_swizzle<new_active_Y,new_active_Z>(), new_layout);
}

} // end namespace detail

template <int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(ComposedLayout<Swizzle<B,M,S>,Offset,Layout> const& layout)
{
  return make_fragment_like(layout.layout_b());
}

//
// Utilities
//

namespace detail {

// Get just the Swizzle part of a composed layout.
template <int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
get_swizzle_portion(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB>)
{
  return Swizzle<B,M,S>{};
}

// A non-swizzled layout's "Swizzle part" is the identity swizzle.
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
get_swizzle_portion(Layout<Shape,Stride>)
{
  return Swizzle<0,4,3>{};
}

// Get the "non-swizzle" part of a composed layout,
// which is the underlying (non-composed) Layout.
template <int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
get_nonswizzle_portion(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& slayout)
{
  return slayout.layout_b();
}

// The non-swizzle part of a non-swizzled layout is just the Layout.
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
get_nonswizzle_portion(Layout<Shape,Stride> const& slayout)
{
  return slayout;
}

} // namespace detail

//
// Slice a Swizzled ComposedLayout
//

namespace detail {

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr
auto
make_swizzle_strides(true_type,
                     IntZ   const& Z,
                     IntY   const& Y,
                     Offset const& offset,
                     int_sequence<I...>)
{
  // Below is an optimized/compressed version of:
  //return cute::make_tuple((swizzle(offset + Z*Int<(1 << I)>{}) - swizzle(offset))...);
  // with knowledge of Swizzle, I... ranges for each B bits,
  //    and the layout won't slice along z-bits that are already set

  // y\z  0   1
  //   0  Z  DC
  //   1 -Z  DC

  return cute::make_tuple(conditional_return((offset & (Y << Int<I>{})) == Int<0>{}, Z * Int<(1 << I)>{}, -Z * Int<(1 << I)>{})...);
}

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr
auto
make_swizzle_strides(false_type,
                     IntZ   const& Z,
                     IntY   const& Y,
                     Offset const& offset,
                     int_sequence<I...>)
{
  // Below is an optimized/compressed version of:
  //return cute::make_tuple((swizzle(offset + Y*Int<(1 << I)>{}) - swizzle(offset))...);
  // with knowledge of Swizzle, I... ranges for each B bits,
  //    and the layout won't slice along y-bits that are already set

  // y\z  0   1
  //   0 Y+Z Y-Z
  //   1 DC  DC

  return cute::make_tuple(conditional_return((offset & (Z << Int<I>{})) == Int<0>{}, (Y+Z) * Int<(1 << I)>{}, (Y-Z) * Int<(1 << I)>{})...);
}

} // end namespace detail

template <class Coord, int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& coord, ComposedLayout<Swizzle<B,M,S>,Offset,Layout> const& layout)
{
  if constexpr (all_underscore<Coord>::value) {
    // Skip the expensive/complicated attempt to decay to a normal layout and just reshape
    return cute::make_tuple(composition(layout.layout_a(), layout.offset(), slice(coord, layout.layout_b())), Int<0>{});
  } else {

    // Projections of the swizzle layout for composition
    auto sw = make_layout(make_shape(Int<(1 << M)>{}, Int<(1 << B)>{}, Int<(1 << (abs(S)-B))>{}, Int<(1 << B)>{}, Int<1>{}));

    auto swizzle_anti_zy = make_layout(shape(sw),
                                       make_stride(stride<0>(sw),      Int<0>{}, stride<2>(sw),      Int<0>{}, size(sw)));
    auto swizzle_only_zy = make_layout(shape(sw),
                                       make_stride(     Int<0>{}, stride<1>(sw),      Int<0>{}, stride<3>(sw), Int<0>{}));

    // The portion of the layout that is not yet consumed
    auto sliced_layout = slice(coord, layout.layout_b());

    // The portion of the layout that we are consuming now
    auto diced_layout = dice(coord, layout.layout_b());
    auto diced_coord  = dice(coord, coord);

    auto diced_layout_anti_zy = composition(swizzle_anti_zy, diced_layout);
    auto diced_layout_only_zy = composition(swizzle_only_zy, diced_layout);

    // New swizzle and offset
    auto swizzle = layout.layout_a();
    // offset_only_zy interacts with swizzle and gets accumulated with layout.offset()
    //   being careful about the static/dynamic contributions from diced_layout and diced_coord
    auto offset_only_zy = layout.offset() ^ to_mixed_bits(diced_layout_only_zy, diced_coord);
    // offset_anti_zy always gets passed through, no interaction with swizzle
    auto offset_anti_zy = diced_layout_anti_zy(diced_coord);

    // If Layout's codomain hits on         Y AND Z, then it's not reducible
    // If Layout's codomain hits on         Y XOR Z, then it's dynamic-normal
    // If Layout's codomain hits on neither Y NOR Z, then it's static-normal

    // If the sliced_layout hits two bits that are swizzled together, then don't attempt to decay

    // Compose with the layout to get the swizzle projection, P o L  [The Z and Y contributing portions of L]
    //   (this also tests that shape/stride of layout compose with swizzle)
    auto sliced_layout_only_zy = composition(swizzle_only_zy, sliced_layout);
    // Transform the end coordinate to get the active bits of the swizzle, (P o L)(c*)
    [[maybe_unused]] auto swizzle_active_bits = sliced_layout_only_zy(size(sliced_layout_only_zy)-Int<1>{});

    // Determine if any active bits collide under the swizzle for potential decay
    if constexpr (is_constant<0, decltype(not (swizzle_active_bits & ~swizzle(swizzle_active_bits)))>::value)
    { // Hits on Y AND Z, so it's not reducible
      return cute::make_tuple(composition(swizzle, offset_only_zy, sliced_layout), offset_anti_zy);
    } else
    { // Misses on Y or Z, so it's static-normal or dynamic-normal

      // Lowest bit of the Z and Y masks
      auto Z = typename Swizzle<B,M,S>::zzz_msk{} & -typename Swizzle<B,M,S>::zzz_msk{};
      auto Y = typename Swizzle<B,M,S>::yyy_msk{} & -typename Swizzle<B,M,S>::yyy_msk{};
      auto stride_lo = detail::make_swizzle_strides(Z < Y, Z, Y, offset_only_zy, make_int_sequence<B>{});
      auto stride_hi = detail::make_swizzle_strides(Z > Y, Z, Y, offset_only_zy, make_int_sequence<B>{});

      // Construct a (dynamic) layout that we can perform the composition with
      auto swizzle_layout = make_layout(make_shape (Int<(1 << M)>{}, repeat<B>(Int<2>{}), Int<(1 << (abs(S)-B))>{}, repeat<B>(Int<2>{}), Int<                  1>{}),
                                        make_stride(Int<       1>{},           stride_lo, Int<(1 <<      (M+B))>{},          stride_hi , Int<(1 << (M+B+abs(S)))>{}));

      // Decay to a normal layout with offset
      return cute::make_tuple(composition(swizzle_layout, sliced_layout),
                              swizzle(offset_only_zy) + offset_anti_zy);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

//
// composition
//

// Ignore identity case
template <int M, int S,
          class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
composition(Swizzle<0,M,S> const&,
            Int<0> const&,
            Layout<Shape,Stride> const& layout)
{
  return layout;
}

template <int B, int M, int S,
          class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
composition(Swizzle<B,M,S> const& sxor,
            Layout<Shape,Stride> const& layout)
{
  return composition(sxor, Int<0>{}, layout);
}

template <class ShapeA, class StrideA,
          int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<ShapeA,StrideA> const& a,
            Swizzle<B,M,S>         const& b)
{
  // Get the Z bits and the Y bits
  auto active_Y = a(typename Swizzle<B,M,S>::yyy_msk{});
  auto active_Z = a(typename Swizzle<B,M,S>::zzz_msk{});

  // Works in simple cases... but could be greatly generalized

  return composition(make_swizzle<active_Y,active_Z>(), a);
}

//
// inverse
//

// Specialization to attempt to pass-through the Swizzle back to the left -- Needed?
template <int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr
auto
right_inverse(ComposedLayout<Swizzle<B,M,S>,Offset,Layout> const& layout)
{
  if constexpr (is_constant<0, Offset>::value) {
    return composition(right_inverse(layout.layout_b()), layout.layout_a());
  } else {
    return composition(right_inverse(layout.layout_b()), right_inverse(layout.offset()), right_inverse(layout.layout_a()));
  }
}

// Specialization to attempt to pass-through the Swizzle back to the left -- Needed?
template <int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr
auto
left_inverse(ComposedLayout<Swizzle<B,M,S>,Offset,Layout> const& layout)
{
  if constexpr (is_constant<0, Offset>::value) {
    return composition(left_inverse(layout.layout_b()), layout.layout_a());
  } else {
    return composition(left_inverse(layout.layout_b()), left_inverse(layout.offset()), left_inverse(layout.layout_a()));
  }
}

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
Swizzle<B,M,S>
right_inverse(Swizzle<B,M,S> const& sw)
{
  return sw;
}

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
Swizzle<B,M,S>
left_inverse(Swizzle<B,M,S> const& sw)
{
  return sw;
}

// Kludge -- Probably want an OffsetFn<T> here instead
template <class T, __CUTE_REQUIRES(is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
right_inverse(T const& t)
{
  return -t;
}

// Kludge -- Probably want an OffsetFn<T> here instead
template <class T, __CUTE_REQUIRES(is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr
auto
left_inverse(T const& t)
{
  return -t;
}

//
// Upcast and Downcast
//

template <int N, int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
upcast(Swizzle<B,M,S> const& swizzle)
{
  static_assert(has_single_bit(N), "N must be a power of two");
  constexpr int log2_n = bit_width(uint32_t(N)) - 1;
  constexpr int NewM   = M - log2_n;
  if constexpr (NewM >= 0) {
    return Swizzle<B,NewM,S>{};
  } else {
    return Swizzle<cute::max(B+NewM,0), 0, S>{};
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
downcast(Swizzle<B,M,S> const& swizzle)
{
  static_assert(has_single_bit(N), "N must be a power of two");
  constexpr int log2_n = bit_width(uint32_t(N)) - 1;
  return Swizzle<B,(M + log2_n),S>{};
}

template <class OldType, class NewType,
          int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
recast_layout(Swizzle<B,M,S> const& swizzle)
{
  using scale = decltype(trait_ratio(sizeof_bits<NewType>{}, sizeof_bits<OldType>{}));
  if constexpr (scale::num == 1 && scale::den == 1) {
    return swizzle;
  }
  else if constexpr (scale::num == 1) {
    return downcast<scale::den>(swizzle);
  }
  else if constexpr (scale::den == 1) {
    return upcast<scale::num>(swizzle);
  }
  else {
    static_assert(dependent_false<scale>, "Recast not supported.");
  }
  CUTE_GCC_UNREACHABLE;
}

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(Swizzle<B,M,S> const&)
{
  return Int<1 << M>{};
}

template <int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& layout)
{
  return gcd(max_alignment(layout.layout_a()),
             max_alignment(layout.offset()),
             max_alignment(layout.layout_b()));
}

//
// Other operations
//

template <int B, int M, int S, class Offset, class LayoutB, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
max_common_layout(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& a,
                  Layout<Shape,Stride>                          const& b)
{
  auto common = max_common_layout(a.layout_b(), b);
  auto base = Int<(1 << M)>{};
  if constexpr (base < size(common)) {
    return common.compose(base);       // Truncate common to size base
  } else {
    return common;
  }
}

template <class Shape, class Stride, int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
max_common_layout(Layout<Shape,Stride>                          const& a,
                  ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& b)
{
  return max_common_layout(b, a);
}

template <int B, int M, int S, class Offset, class LayoutB, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& a,
                  Layout<Shape,Stride>                          const& b)
{
  // This assumes that Offset is in the YZ domain of the Swizzle...
  return cute::min(max_common_vector(a.layout_b(), b), Int<(1 << M)>{});
}

template <class Shape, class Stride, int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(Layout<Shape,Stride>                          const& a,
                  ComposedLayout<Swizzle<B,M,S>,Offset,LayoutB> const& b)
{
  return max_common_vector(b, a);
}

template <int B0, int M0, int S0, class Offset0, class LayoutB0,
          int B1, int M1, int S1, class Offset1, class LayoutB1>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(ComposedLayout<Swizzle<B0,M0,S0>,Offset0,LayoutB0> const& a,
                  ComposedLayout<Swizzle<B1,M1,S1>,Offset1,LayoutB1> const& b)
{
  // Typical impl is composition(a, right_inverse(b))
  // so this is  Sw0 o B0 o rinv(Sw1 o B1) = Sw0 o B0 o rinv(B1) o Sw1
  auto vec = max_common_vector(a.layout_b(), b.layout_b());

  // This assumes that Offset is in the YZ domain of the Swizzle...
  if constexpr (Swizzle<B0,M0,S0>{} == Swizzle<B1,M1,S1>{}) {
    return vec;
  } else {
    return cute::min(vec, Int<(1 << M0)>{}, Int<(1 << M1)>{});
  }

  CUTE_GCC_UNREACHABLE;
}

///////////////////////////////////////////////////////////////////////////////
// ComposedLayout as second argument is often more difficult...

template <class Shape, class Stride,
          int B, int M, int S, class Offset, class LayoutT>
CUTE_HOST_DEVICE constexpr
auto
logical_product(Layout<Shape,Stride>                          const& layout,
                ComposedLayout<Swizzle<B,M,S>,Offset,LayoutT> const& tiler)
{
  CUTE_STATIC_ASSERT_V(tiler.offset() == Int<0>{}, "Require Swizzle offset == 0.");
  // The new layout -- if swizzle wasn't an issue, this is the result
  //   our goal is to determine a new swizzle for these strides
  auto new_layout = logical_product(layout, tiler.layout_b());

  // This is accomplished by identifying
  //  S o L  :=:  S? o L*
  // We identify the "active" portion of S by computing (P o L)(c*) where P is a projection generated by S
  // Then that active identifier is transformed through the layouts:
  //  L*(L[(P o L)(c*)])
  // which is a new swizzle identifier for S?, the new swizzle

  // Projections of the swizzle layout for composition, P
  auto swizzle_only_zy = make_layout(make_shape (Int<(1 << M)>{}, Int<(1 << B)>{}, Int<(1 << (abs(S)-B))>{}, Int<(1 <<  B        )>{}, Int<1>{}),
                                     make_stride(       Int<0>{}, Int<(1 << M)>{},                 Int<0>{}, Int<(1 << (M+abs(S)))>{}, Int<0>{}));

  // Compose with the tiler to get the swizzle projection, P o L  [The Z and Y contributing portions of L]
  auto layout_only_zy       = composition(swizzle_only_zy, tiler.layout_b());
  // Transform the end coordinate to get the active bits of the swizzle, (P o L)(c*)
  auto swizzle_active_bits  = layout_only_zy(size(layout_only_zy)-Int<1>{});
  // Get the Z bit and the Y bits
  auto active_Z = swizzle_active_bits & typename Swizzle<B,M,S>::zzz_msk{};
  auto active_Y = swizzle_active_bits & typename Swizzle<B,M,S>::yyy_msk{};

  // Pass the identifiers through the old layout and new layout to make a new swizzle identifier, L*(L[(P o L)(c*)])
  auto new_active_Z = new_layout(Int<0>{}, tiler.layout_b()[active_Z]);
  auto new_active_Y = new_layout(Int<0>{}, tiler.layout_b()[active_Y]);

  // Use this new swizzle identifier to construxt the new swizzle for new_layout
  //   (this also makes sure it's a "valid" swizzle that Swizzle can represent)
  return composition(make_swizzle<new_active_Y,new_active_Z>(), new_layout);
}

} // end namespace cute
