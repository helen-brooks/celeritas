#!/bin/sh -e

OS=$1
CMAKE_PRESET=$2

set -x
test -n "${OS}"
test -n "${CMAKE_PRESET}"

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR="$(dirname $0)"/../..
fi
cd "${CELER_SOURCE_DIR}"
CELER_SOURCE_DIR=$(pwd)
ln -fs scripts/cmake-presets/ci-${OS}.json CMakeUserPresets.json

# Source environment script if necessary
_ENV_SCRIPT="scripts/env/ci-${OS}.sh"
if [ -f "${_ENV_SCRIPT}" ]; then
  . "${_ENV_SCRIPT}"
fi

# Fetch tags for version provenance
git fetch --tags
# Configure
cmake --preset=${CMAKE_PRESET}
# Build and (optionally) install
cmake --build --preset=${CMAKE_PRESET}

# Require regression-like tests to be enabled and pass
export CELER_TEST_STRICT=1

# Set up test arguments
CTEST_TOOL=Test
CTEST_ARGS=""
case ${CMAKE_PRESET} in
  vecgeom-demos )
    CTEST_ARGS="-L app"
    ;;
  valgrind )
    CTEST_TOOL="MemCheck"
    CTEST_ARGS="-LE nomemcheck"
    ;;
  * )
    ;;
esac

# Run tests
cd build
ctest -T ${CTEST_TOOL} ${CTEST_ARGS}\
  -j16 --timeout 180 \
  --no-compress-output --output-on-failure \
  --test-output-size-passed=65536 --test-output-size-failed=1048576 \
# List XML files generated: jenkins will upload these later
find Testing -name '*.xml'
  
# Test examples against installed celeritas
export CMAKE_PRESET
export CELER_SOURCE_DIR
if [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  export LDFLAGS=-Wl,--no-as-needed # for Ubuntu with vecgeom?
  exec ${CELER_SOURCE_DIR}/scripts/ci/test-examples.sh
elif [ "${CMAKE_PRESET}" = "full-novg-ndebug" ] \
  || [ "${CMAKE_PRESET}" = "hip-ndebug" ]  ; then
  exec ${CELER_SOURCE_DIR}/scripts/ci/test-examples.sh
fi
