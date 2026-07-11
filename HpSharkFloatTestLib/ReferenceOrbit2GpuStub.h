#pragma once

template <class SharkFloatParams> struct HpSharkReferenceResults;

namespace HpShark {

// Launches a test-only CUDA kernel that writes the intentional Ref2 sentinel to
// the reference-orbit result supplied by a GpuOrbitSession.
template <class SharkFloatParams>
void InvokeReferenceOrbit2GpuStub(HpSharkReferenceResults<SharkFloatParams> &combo);

} // namespace HpShark
