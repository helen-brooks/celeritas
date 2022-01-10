//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.i.hh
//---------------------------------------------------------------------------//
#include <cmath>

#include "base/Algorithms.hh"
#include "base/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and the stepper.
 */
template<class StepperT>
CELER_FUNCTION FieldDriver<StepperT>::FieldDriver(const FieldParamsData& shared,
                                                  StepperT* stepper)
    : shared_(shared), stepper_(*stepper)
{
    CELER_EXPECT(shared_ && stepper);
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control based on G4ChordFinder and G4MagIntegratorDriver.
 *
 * \param step maximum step length
 * \param state starting state
 * \return substep and updated state
 *
 * For a given trial step, advance by a sub-step within a required tolerance
 * and update the current state (position and momentum).  For an efficient
 * adaptive integration, the proposed chord of which the miss-distance (the
 * closest distance from the curved trajectory to the chord) is smaller than
 * a reference distance (dist_chord) will be accepted if its stepping error is
 * within a reference accuracy. Otherwise, the more accurate step integration
 * (advance_accurate) will be performed.
 */
template<class StepperT>
CELER_FUNCTION DriverResult
FieldDriver<StepperT>::advance(real_type step, const OdeState& state)
{
    CELER_EXPECT(step >= this->minimum_step());

    // Output with a step control error
    ChordSearch output = this->find_next_chord(step, state);

    // Evaluate the relative error
    real_type rel_error = output.error
                          / (shared_.epsilon_step * output.end.step);

    if (rel_error > 1)
    {
        // Advance more accurately with a newly proposed step
        real_type next_step = this->new_step_size(step, rel_error);
        output.end          = this->accurate_advance(
            output.end.step, output.end.state, next_step);
    }

    return output.end;
}

//---------------------------------------------------------------------------//
/*!
 * Find the next acceptable chord of which the miss-distance is smaller than
 * a given reference (delta_chord) and evaluate the associated error.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::find_next_chord(real_type step, const OdeState& state)
    -> ChordSearch
{
    // Output with a step control error
    ChordSearch output;

    bool          succeeded       = false;
    unsigned int  remaining_steps = shared_.max_nsteps;
    StepperResult result;

    do
    {
        // Try with the proposed step
        result = stepper_(step, state);

        // Check whether the distance to the chord is small than the reference
        real_type dchord = detail::distance_chord(
            state, result.mid_state, result.end_state);

        if (dchord <= (shared_.delta_chord + FieldDriver::ppm()))
        {
            succeeded    = true;
            output.error = detail::truncation_error(
                step, shared_.epsilon_rel_max, state, result.err_state);
        }
        else
        {
            // Estimate a new trial chord with a relative scale
            step *= std::fmax(std::sqrt(shared_.delta_chord / dchord), half());
        }
    } while (!succeeded && (--remaining_steps > 0));

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(succeeded);

    // Update step, position and momentum
    output.end.step  = step;
    output.end.state = result.end_state;

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Accurate advance for an adaptive step control.
 *
 * Perform an adaptive step integration for a proposed step or a series of
 * sub-steps within a required tolerance until the the accumulated curved path
 * is equal to the input step length.
 */
template<class StepperT>
CELER_FUNCTION DriverResult FieldDriver<StepperT>::accurate_advance(
    real_type step, const OdeState& state, real_type hinitial)
{
    CELER_ASSERT(step > 0);

    // Set an initial proposed step and evaluate the minimum threshold
    real_type end_curve_length = step;

    // Use a pre-defined initial step size if it is smaller than the input
    // step length and larger than the permillion fraction of the step length.
    // Otherwise, use the input step length for the first trial.
    // TODO: review whether this approach is an efficient bootstrapping.
    real_type h = ((hinitial > FieldDriver::ppm() * step) && (hinitial < step))
                      ? hinitial
                      : step;
    real_type h_threshold = shared_.epsilon_step * step;

    // Output with the next good step
    Integration output;
    output.end.state = state;

    // Performance integration
    bool         succeeded       = false;
    real_type    curve_length    = 0;
    unsigned int remaining_steps = shared_.max_nsteps;

    do
    {
        output = this->integrate_step(h, output.end.state);

        curve_length += output.end.step;

        if (h < h_threshold || curve_length >= end_curve_length)
        {
            succeeded = true;
        }
        else
        {
            h = std::fmax(std::fmax(output.proposed_step, shared_.minimum_step),
                          end_curve_length - curve_length);
        }
    } while (!succeeded && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(succeeded);

    output.end.step = curve_length;
    return output.end;
}

//---------------------------------------------------------------------------//
/*!
 * Advance for a given step and evaluate the next predicted step.
 *
 * Helper function for accurate_advance.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::integrate_step(real_type step, const OdeState& state)
    -> Integration
{
    // Output with a next proposed step
    Integration output;

    if (step > shared_.minimum_step)
    {
        output = this->one_good_step(step, state);
    }
    else
    {
        // Do an integration step for a small step (a.k.a quick advance)
        StepperResult result = stepper_(step, state);

        // Update position and momentum
        output.end.state = result.end_state;

        real_type dyerr = detail::truncation_error(
            step, shared_.epsilon_rel_max, state, result.err_state);
        output.end.step = step;

        // Compute a proposed new step
        CELER_ASSERT(output.end.step > 0);
        output.proposed_step
            = this->new_step_size(step, dyerr / (step * shared_.epsilon_step));
    }

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Advance within a relative truncation error and estimate a good step size
 * for the next integration.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::one_good_step(real_type step, const OdeState& state)
    -> Integration
{
    // Output with a proposed next step
    Integration output;

    // Perform integration for adaptive step control with the trunction error
    bool          succeeded       = false;
    unsigned int  remaining_steps = shared_.max_nsteps;
    real_type     errmax2 = celeritas::numeric_limits<real_type>::max();
    StepperResult result;

    do
    {
        result  = stepper_(step, state);
        errmax2 = detail::truncation_error(
            step, shared_.epsilon_rel_max, state, result.err_state);

        if (errmax2 <= 1)
        {
            succeeded = true;
        }
        else
        {
            // Step failed; compute the size of re-trial step.
            real_type htemp = shared_.safety * step
                              * std::pow(errmax2, half() * shared_.pshrink);

            // Truncation error too large, reduce stepsize with a low bound
            step = std::fmax(htemp, shared_.max_stepping_decrease * step);
        }
    } while (!succeeded && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(succeeded);

    // Update state, step taken by this trial and the next predicted step
    output.end.state     = result.end_state;
    output.end.step      = step;
    output.proposed_step = (errmax2 > ipow<2>(shared_.errcon))
                               ? shared_.safety * step
                                     * std::pow(errmax2, half() * shared_.pgrow)
                               : shared_.max_stepping_increase * step;

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Estimate the new predicted step size based on the error estimate.
 */
template<class StepperT>
CELER_FUNCTION real_type
               FieldDriver<StepperT>::new_step_size(real_type step, real_type rel_error) const
{
    CELER_ASSERT(rel_error > 0);
    real_type scale_factor
        = std::pow(rel_error, rel_error > 1 ? shared_.pshrink : shared_.pgrow);
    return shared_.safety * step * scale_factor;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
