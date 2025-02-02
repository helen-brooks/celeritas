//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/PolyEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Functor class to evaluate a polynomial.
 *
 * This is an efficient and foolproof way of storing and evaluating a
 * polynomial expansion:
 * \f[
  f(x) = a_0 + x * (a_1 + x * (a_2 + ...))
  \f]
 *
 * It replaces opaque expressions such as:
 * \code
    corr = (zeff * (real_type(1.84035e-4) * zeff - real_type(1.86427e-2))
             + real_type(1.41125));
 * \endcode
 * with
 * \code
    corr = make_poly_evaluator(1.41125, -1.86427e-2, 1.84035e-4)(zeff);
   \endcode
 * or, to use an explicit type without having to cast each coefficient:
 * \code
   using PolyQuad = PolyEvaluator<real_type, N>;
   corr = PolyQuad{1.41125, -1.86427e-2, 1.84035e-4)(zeff);
 * \endcode
 */
template<class T, unsigned int N>
class PolyEvaluator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = T;
    using argument_type = T;
    using ArrayT = Array<T, N + 1>;
    //!@}

  public:
    //! Construct with the polynomial to evaluate
    template<class... Ts>
    explicit CELER_CONSTEXPR_FUNCTION PolyEvaluator(Ts... coeffs)
        : coeffs_{static_cast<T>(coeffs)...}
    {
        // Protect against leaving off a coefficient, e.g. PolyQuad(1, 2)
        static_assert(sizeof...(coeffs) == N + 1,
                      "All coefficients for PolyEvaluator must be explicitly "
                      "specified");
    }

    //! Construct from an array of data
    CELER_CONSTEXPR_FUNCTION PolyEvaluator(ArrayT const& coeffs)
        : coeffs_{coeffs}
    {
    }

    //! Evaluate the polynomial at the given value
    CELER_CONSTEXPR_FUNCTION T operator()(T arg) const
    {
        return this->calc_impl<0>(arg);
    }

  private:
    const ArrayT coeffs_;

    template<unsigned int M, std::enable_if_t<(M < N), int> = 0>
    CELER_CONSTEXPR_FUNCTION T calc_impl(T arg) const
    {
        return coeffs_[M] + arg * calc_impl<M + 1>(arg);
    }

    template<unsigned int M, std::enable_if_t<(M == N), int> = 0>
    CELER_CONSTEXPR_FUNCTION T calc_impl(T) const
    {
        return coeffs_[N];
    }
};

//---------------------------------------------------------------------------//
/*!
 * Create a polynomial evaluator from the given arguments.
 */
template<typename... Ts>
constexpr auto make_poly_evaluator(Ts... args)
{
    using value_type = std::common_type_t<Ts...>;
    return PolyEvaluator<value_type, sizeof...(Ts) - 1>{args...};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
