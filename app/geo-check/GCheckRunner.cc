//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckRunner.cc
//---------------------------------------------------------------------------//
#include "GCheckRunner.hh"

#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>
#include <math.h>

#include "corecel/Types.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Logger.hh"
#include "orange/OrangeData.hh"
#include "celeritas/geo/GeoData.hh"

#include "GCheckKernel.hh"

using namespace celeritas;

namespace geo_check
{
//---------------------------------------------------------------------------//
/*!
 * Constructor, takes ownership of SPConstGeo object received
 */
GCheckRunner::GCheckRunner(SPConstGeo const& geometry, int max_steps, bool cuda)
    : geo_params_(std::move(geometry)), max_steps_(max_steps), use_cuda_(cuda)
{
    CELER_EXPECT(geo_params_);
    CELER_EXPECT(max_steps > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a track for debugging purposes
 */
void GCheckRunner::operator()(GeoTrackInitializer const* gti) const
{
    using StateStore = CollectionStateStore<GeoStateData, MemSpace::device>;

    CELER_EXPECT(gti);

    // run on the CPU
    CELER_LOG(status) << "Propagating track(s) on CPU";
    auto cpudata = run_cpu(geo_params_, gti, this->max_steps_);

    // skip GPU if not available
    if (!use_cuda_)
        return;

    // run on the GPU
    GCheckInput input;
    input.init.push_back(*gti);
    input.max_steps = this->max_steps_;
    input.params = this->geo_params_->device_ref();

    StateStore states(this->geo_params_->host_ref(), 1);
    input.state = states.ref();

    CELER_LOG(status) << "Propagating track(s) on GPU";
    auto gpudata = run_gpu(input);

    CELER_LOG(status) << " # steps: cpu=" << cpudata.distances.size()
                      << " gpu=" << gpudata.distances.size() << "\n";
    CELER_ASSERT(cpudata.ids.size() == gpudata.ids.size());

    if (!(cpudata.ids == gpudata.ids && cpudata.distances == gpudata.distances))
    {
        for (size_type i = 0; i < cpudata.distances.size(); ++i)
        {
            if (cpudata.ids[i] != gpudata.ids[i]
                || fabs(cpudata.distances[i] - gpudata.distances[i]) > 1.0e-12)
            {
                std::cout
                    << "compare i=" << i << ": ids=" << cpudata.ids[i] << ", "
                    << gpudata.ids[i] << "; cpudist=" << cpudata.distances[i]
                    << ", " << gpudata.distances[i] << " , diff="
                    << cpudata.distances[i] - gpudata.distances[i] << "\n";
            }
        }
    }
    CELER_ASSERT(cpudata.ids == gpudata.ids);
    CELER_LOG(status) << "Comparison successful.";
}
//---------------------------------------------------------------------------//

}  // namespace geo_check
