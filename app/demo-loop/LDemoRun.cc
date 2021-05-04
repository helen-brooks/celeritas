//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoRun.cc
//---------------------------------------------------------------------------//
#include "LDemoRun.hh"

#include "base/CollectionStateStore.hh"
#include "comm/Logger.hh"
#include "physics/base/ModelInterface.hh"
#include "LDemoParams.hh"
#include "LDemoInterface.hh"
#include "LDemoKernel.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
template<class P, MemSpace M>
struct ParamsGetter;

template<class P>
struct ParamsGetter<P, MemSpace::host>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.host_pointers();
    }
};

template<class P>
struct ParamsGetter<P, MemSpace::device>
{
    const P& params_;

    auto operator()() const -> decltype(auto)
    {
        return params_.device_pointers();
    }
};

template<MemSpace M, class P>
decltype(auto) get_pointers(const P& params)
{
    return ParamsGetter<P, M>{params}();
}

//---------------------------------------------------------------------------//
template<MemSpace M>
ParamsData<Ownership::const_reference, M>
build_params_refs(const LDemoParams& p)
{
    ParamsData<Ownership::const_reference, M> ref;
    ref.geometry  = get_pointers<M>(*p.geometry);
    ref.materials = get_pointers<M>(*p.materials);
    ref.geo_mats  = get_pointers<M>(*p.geo_mats);
    ref.cutoffs   = get_pointers<M>(*p.cutoffs);
    ref.particles = get_pointers<M>(*p.particles);
    ref.physics   = get_pointers<M>(*p.physics);
    ref.rng       = get_pointers<M>(*p.rng);
    CELER_ENSURE(ref);
    return ref;
}

//---------------------------------------------------------------------------//
/*!
 * Launch interaction kernels for all applicable models.
 *
 * For now, just launch *all* the models.
 */
void launch_models(LDemoParams const     host_params,
                   ParamsDeviceRef const params,
                   StateDeviceRef const  states)
{
    CELER_NOT_IMPLEMENTED("TODO: add remaining processes");
    // TODO: for this to work on host, we'll need to template
    // ModelInterface on MemSpace and overload the `interact`
    // method on Model to work with device pointers.

    // Create ModelInteractPointers
    ModelInteractPointers pointers;
    pointers.params.particle = params.particles;
    pointers.params.material = params.materials;
    pointers.params.physics  = params.physics;
    pointers.states.particle = states.particles;
    pointers.states.material = states.materials;
    pointers.states.physics  = states.physics;
    pointers.states.rng      = states.rng;
    // TODO: direction
    pointers.secondaries = states.secondaries;
    pointers.result
        = states.interactions[AllItems<Interaction, MemSpace::device>{}];
    CELER_ASSERT(pointers);

    // Loop over physics models IDs and invoke `interact`
    for (auto model_id : range(ModelId{host_params.physics->num_models()}))
    {
        const Model& model = host_params.physics->model(model_id);
        model.interact(pointers);
    }
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
LDemoResult run_gpu(LDemoArgs args)
{
    CELER_EXPECT(args);

    // Load all the problem data
    LDemoParams params = load_params(args);

    // Create param interfaces (TODO unify with sim/TrackInterface )
    ParamsDeviceRef params_ref = build_params_refs<MemSpace::device>(params);

    // Create states (TODO state store?)
    StateData<Ownership::value, MemSpace::device> state_storage;
    resize(&state_storage,
           build_params_refs<MemSpace::host>(params),
           args.num_tracks);
    StateDeviceRef states_ref = make_ref(state_storage);

    CELER_NOT_IMPLEMENTED("TODO: stepping loop");

    // TODO: Initialize fixed number of primaries (isotropic samples?)

    bool any_alive = true;
    while (any_alive)
    {
        demo_loop::pre_step(params_ref, states_ref);
        demo_loop::along_and_post_step(params_ref, states_ref);
        launch_models(params, params_ref, states_ref);
        demo_loop::process_interactions(params_ref, states_ref);
        // TODO: Create primaries from secondaries
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
