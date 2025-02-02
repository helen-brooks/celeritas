//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "CutoffData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant material- and particle-dependent cutoff values.
 *
 * \c CutoffParamsData is defined in \c CutoffData.hh and constructed by
 * \c CutoffParams .
 *
 * \code
 * CutoffParams cutoffs(input);
 * CutoffView cutoff_view(cutoffs.host_ref(), material_id);
 * cutoff_view.energy(particle_id);
 * cutoff_view.range(particle_id);
 * \endcode
 */
class CutoffView
{
  public:
    //!@{
    //! \name Type aliases
    using CutoffId = OpaqueId<ParticleCutoff>;
    using CutoffData = NativeCRef<CutoffParamsData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct for the given particle and material ids
    inline CELER_FUNCTION
    CutoffView(CutoffData const& params, MaterialId material);

    // Return energy cutoff value
    inline CELER_FUNCTION Energy energy(ParticleId particle) const;

    // Return range cutoff value
    inline CELER_FUNCTION real_type range(ParticleId particle) const;

  private:
    CutoffData const& params_;
    MaterialId material_;

    //// HELPER FUNCTIONS ////

    // Get the cutoff for the given particle and material
    CELER_FORCEINLINE_FUNCTION ParticleCutoff get(ParticleId particle) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct view from host/device for the given material id.
 */
CELER_FUNCTION
CutoffView::CutoffView(CutoffData const& params, MaterialId material)
    : params_(params), material_(material)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(material_ < params_.num_materials);
}

//---------------------------------------------------------------------------//
/*!
 * Return energy cutoff value.
 */
CELER_FUNCTION auto CutoffView::energy(ParticleId particle) const -> Energy
{
    return this->get(particle).energy;
}

//---------------------------------------------------------------------------//
/*!
 * Return range cutoff value.
 */
CELER_FUNCTION real_type CutoffView::range(ParticleId particle) const
{
    return this->get(particle).range;
}

//---------------------------------------------------------------------------//
/*!
 * Get the cutoff for the given particle and material.
 */
CELER_FUNCTION ParticleCutoff CutoffView::get(ParticleId particle) const
{
    CELER_EXPECT(particle < params_.id_to_index.size());
    CELER_EXPECT(params_.id_to_index[particle] < params_.num_particles);
    CutoffId id{params_.num_materials * params_.id_to_index[particle]
                + material_.get()};
    CELER_ENSURE(id < params_.cutoffs.size());
    return params_.cutoffs[id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
