//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldMapData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * FieldMap (2-dimensional RZ map) parameters
 */
struct FieldMapParameters
{
    size_type num_grid_r;
    size_type num_grid_z;
    real_type delta_grid;
    real_type offset_z;
};

//---------------------------------------------------------------------------//
/*!
 * FieldMap element
 */
struct FieldMapElement
{
    float value_z;
    float value_r;
};

//---------------------------------------------------------------------------//
/*!
 * FieldMap input data.
 *
 * A vector of size [num_grid_z*num_grid_r] which stores data
 * for the equivalent 2-dimensional RZ-array[num_grid_z][num_grid_r] and
 * associated parameters
 */
struct FieldMapInput
{
    FieldMapParameters params;
    std::vector<FieldMapElement> data;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct FieldMapData
{
    //! Parameters of FieldMap
    FieldMapParameters params;

    //! Index of FieldMap Collection
    using ElementId = ItemId<size_type>;

    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    ElementItems<FieldMapElement> fieldmap;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !fieldmap.empty();
    }

    inline CELER_FUNCTION bool valid(size_type idx_z, size_type idx_r) const
    {
        return (idx_z < params.num_grid_z && idx_r < params.num_grid_r);
    }

    inline CELER_FUNCTION ElementId id(int idx_z, int idx_r) const
    {
        return ElementId(idx_z * params.num_grid_r + idx_r);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    FieldMapData& operator=(FieldMapData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        params = other.params;
        fieldmap = other.fieldmap;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
