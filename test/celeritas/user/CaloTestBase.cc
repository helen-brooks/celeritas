//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.cc
//---------------------------------------------------------------------------//
#include "CaloTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/StepCollector.hh"

#include "ExampleCalorimeters.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
CaloTestBase::~CaloTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct example callback and step collector at setup time.
 */
void CaloTestBase::SetUp()
{
    example_calos_ = std::make_shared<ExampleCalorimeters>(
        *this->geometry(), this->get_detector_names());

    this->num_detectors_= this->get_detector_names().size();

    StepCollector::VecInterface interfaces = {example_calos_};

    collector_ = std::make_shared<StepCollector>(
        std::move(interfaces), this->geometry(), this->action_reg().get());
}

//---------------------------------------------------------------------------//
//! Print the expected result
void CaloTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const double expected_edep[] = "
         << repr(this->edep)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
auto CaloTestBase::run(size_type num_tracks,
                       size_type num_steps_per_batch,
                       size_type num_batches) -> RunResult
{
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.num_track_slots = num_tracks;

    // Create a stepper from the input
    Stepper<MemSpace::host> step(step_inp);

    // Can't have fewer than 1 track per batch
    CELER_ASSERT(num_batches<=num_tracks);

    // Don't want to deal with remainders
    CELER_ASSERT(num_tracks%num_batches==0);

    // Compute tracks per batch
    size_type num_tracks_per_batch=num_tracks/num_batches;

    // Initialize RunResult
    RunResult result;
    result.edep=std::vector<double>(this->num_detectors_,0.);
    result.edep_err=std::vector<double>(this->num_detectors_,0.);

    // Loop over batches
    for(size_type i_batch=0; i_batch<num_batches; ++i_batch){

      // Get a vector of primary particles
      auto primaries = this->make_primaries(num_tracks_per_batch);

      // Initial step
      auto count = step(make_span(primaries));

      // This loop will perform num_steps, or stop if we've exhasted all
      // the primaries
      while (count && --num_steps_per_batch > 0){
        count = step();
      }

      // Retrieve energies deposited this batch for each detector
      auto edep = example_calos_->deposition();

      // Update results for each detector
      for(size_t i_det=0; i_det<this->num_detectors_; ++i_det){
        auto edep_det=edep[i_det];
        result.edep.at(i_det)+=edep_det;
        result.edep_err.at(i_det)+=(edep_det*edep_det);
      }

      example_calos_->clear();
    }

    // Finally, compute the mean and relative_err
    double norm=num_batches > 1 ?  1.0/double(num_batches) : 1.0;
    for(size_t i_det=0; i_det<this->num_detectors_; ++i_det){
      auto mu=result.edep.at(i_det)*norm;
      auto var=result.edep_err.at(i_det)*norm - mu*mu;
      auto err=sqrt(var) / mu;
      result.edep.at(i_det) = mu;
      result.edep_err.at(i_det) = err;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
