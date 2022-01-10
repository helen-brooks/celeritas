//----------------------------------*-cu-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CleanupKernel.cu
//! \note Auto-generated by gen-demo-loop-kernel.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Types.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "../LDemoLauncher.hh"

using namespace celeritas;

namespace demo_loop
{
namespace generated
{
namespace
{
__global__ void cleanup_kernel(
    ParamsDeviceRef const params,
    StateDeviceRef const states)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < 1))
        return;

    CleanupLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}
} // namespace

void cleanup(
    const celeritas::ParamsDeviceRef& params,
    const celeritas::StateDeviceRef& states)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);

    static const KernelParamCalculator cleanup_ckp(
        cleanup_kernel, "cleanup");
    auto kp = cleanup_ckp(1);
    cleanup_kernel<<<kp.grid_size, kp.block_size>>>(
        params, states);
    CELER_CUDA_CHECK_ERROR();
}

} // namespace generated
} // namespace demo_loop
