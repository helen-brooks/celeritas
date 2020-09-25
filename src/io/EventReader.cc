//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EventReader.cc
//---------------------------------------------------------------------------//
#include "EventReader.hh"
#include "physics/base/Units.hh"
#include "HepMC3/GenEvent.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a filename.
 */
EventReader::EventReader(const char* filename, constSPParticleParams params)
    : params_(std::move(params))
{
    // Determine the input file format and construct the appropriate reader
    input_file_ = HepMC3::deduce_reader(filename);
    ENSURE(input_file_);
}

//---------------------------------------------------------------------------//
/*!
 * Read the primary particles from the event record.
 */
EventReader::result_type EventReader::operator()()
{
    result_type result;
    int         event_id = -1;

    while (!input_file_->failed())
    {
        // Parse the next event from the record
        HepMC3::GenEvent gen_event;
        input_file_->read_event(gen_event);

        // There are no more events
        if (input_file_->failed())
        {
            break;
        }
        ++event_id;

        // Convert the energy units to MeV and the length units to cm
        gen_event.set_units(HepMC3::Units::MEV, HepMC3::Units::CM);

        for (auto gen_particle : gen_event.particles())
        {
            Primary primary;

            // Set the event number
            primary.event_id = EventId(event_id);

            // Get the position of the primary
            auto pos         = gen_event.event_pos();
            primary.position = {pos.x() * units::centimeter,
                                pos.y() * units::centimeter,
                                pos.z() * units::centimeter};

            // Calculate the magnitude of the momentum
            real_type momentum = gen_particle->momentum().length()
                                 * units::mega_electron_volt
                                 / units::speed_of_light;

            // Get the direction of the primary
            primary.direction = {gen_particle->momentum().px() / momentum,
                                 gen_particle->momentum().py() / momentum,
                                 gen_particle->momentum().pz() / momentum};

            // Get the registered ID of the particle from the PDG code
            int pdg        = gen_particle->pid();
            primary.def_id = params_->find(PDGNumber(pdg));

            // Get the energy of the primary
            primary.energy = gen_particle->momentum().e()
                             * units::mega_electron_volt;

            result.push_back(primary);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
