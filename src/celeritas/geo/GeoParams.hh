//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoParams.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_VECGEOM
#    include "celeritas/ext/VecgeomParams.hh"  // IWYU pragma: export
#else
#    include "orange/OrangeParams.hh"  // IWYU pragma: export
#endif

// Include type alias for GeoParams
#include "GeoParamsFwd.hh"  // IWYU pragma: export
