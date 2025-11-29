#pragma once
template <class SharkFloatParams>
constexpr int32_t
CalculateMultiplySharedMemorySize()
{
    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2; // Half of NewN

    // Figure out how much shared memory to allocate if we're not loading
    // everything into shared memory and instead using a constant amount.
    constexpr auto sharedRequired = SharkFloatParams::GlobalThreadsPerBlock * sizeof(uint64_t) * 3;

    // HpShark::ConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes =
        HpShark::LoadAllInShared ? (2 * NewN + 2 * n) * sizeof(uint32_t) : sharedRequired;

    return sharedAmountBytes;
}