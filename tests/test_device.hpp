#pragma once

// -----------------------------------------------------------------------------
// The test device source.
//
// Every device-using gate reaches its device through this header instead of a
// production heuristic, so a CI run can pin the device with the environment
// (`AEON_TEST_DEVICE_IDS`, exported by `aeon_add_test`) and a multi-device gate can
// read the same list it was handed. The default is one device, id 0.
// -----------------------------------------------------------------------------

#include "infrastructure/parallel/parallel_topology.hpp"
#include "platform/device.hpp"

#include <cstdlib>
#include <string>
#include <vector>

namespace aeon::test {

// The ids from `AEON_TEST_DEVICE_IDS` (a `parse_device_ids` list), or `{0}` when
// unset. A multi-device gate reads the full list; a single-device gate takes the
// first entry.
inline std::vector<int> test_device_ids() {
    const char* env = std::getenv("AEON_TEST_DEVICE_IDS");
    if (env == nullptr || *env == '\0') return {0};
    return aeon::core::parse_device_ids(env);
}

// Select and return the first id. This is the replacement for the deleted
// `select_compute_device`: the choice is the environment's, not a bus scan's.
inline int select_test_device(bool verbose = true) {
    const std::vector<int> ids = test_device_ids();
    return aeon::core::select_device(ids.empty() ? 0 : ids.front(), verbose);
}

} // namespace aeon::test
