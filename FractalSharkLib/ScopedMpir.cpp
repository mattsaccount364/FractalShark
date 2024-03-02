#include "stdafx.h"

#include "ScopedMpir.h"
#include "HighPrecision.h"

thread_local uint8_t ScopedMPIRAllocators::Allocated[NumBlocks][BytesPerBlock];
thread_local size_t ScopedMPIRAllocators::AllocatedSize[NumBlocks];
thread_local size_t ScopedMPIRAllocators::AllocatedIndex = 0;
thread_local size_t ScopedMPIRAllocators::AllocationsAndFrees[NumBlocks];

ScopedMPIRAllocators::ScopedMPIRAllocators() :
    ExistingMalloc{},
    ExistingRealloc{},
    ExistingFree{} {
}

ScopedMPIRAllocators::~ScopedMPIRAllocators() {
    if (ExistingMalloc == nullptr) {
        return;
    }

    mp_set_memory_functions(
        ExistingMalloc,
        ExistingRealloc,
        ExistingFree);
}

void ScopedMPIRAllocators::InitScopedAllocators() {
    mp_get_memory_functions(
        &ExistingMalloc,
        &ExistingRealloc,
        &ExistingFree);

    mp_set_memory_functions(
        NewMalloc,
        NewRealloc,
        NewFree);
}

void ScopedMPIRAllocators::InitTls() {
    memset(Allocated, 0, sizeof(Allocated));
    memset(AllocatedSize, 0, sizeof(AllocatedSize));
    AllocatedIndex = 0;
    memset(AllocationsAndFrees, 0, sizeof(AllocationsAndFrees));
}

void ScopedMPIRAllocators::ShutdownTls() {
    // If there are any allocations left, we have a memory leak.
    for (size_t i = 0; i < NumBlocks; i++) {
        if (AllocationsAndFrees[i] != 0) {
            DebugBreak();
        }
    }
}

void* ScopedMPIRAllocators::NewMalloc(size_t size) {
    const auto alignedSize = (size + 63) & ~63;
    size_t indexToAllocateAt = AllocatedSize[AllocatedIndex];

    if (AllocatedSize[AllocatedIndex] + alignedSize >= BytesPerBlock) {
        // Find the next block with no allocations:
        for (;;) {
            AllocatedIndex = (AllocatedIndex + 1) % NumBlocks;
            if (AllocationsAndFrees[AllocatedIndex] == 0) {
                AllocatedSize[AllocatedIndex] = 0;
                indexToAllocateAt = 0;
                break;
            }
        }
    }
    else {
        indexToAllocateAt = AllocatedSize[AllocatedIndex];
    }

    auto ret = &Allocated[AllocatedIndex][AllocatedSize[AllocatedIndex]];

    // Increment AllocatedSize by 64-byte aligned size:
    AllocatedSize[AllocatedIndex] += alignedSize;
    AllocationsAndFrees[AllocatedIndex]++;

    return ret;
}

void* ScopedMPIRAllocators::NewRealloc(void* ptr, size_t old_size, size_t new_size) {
    // This one is like new_malloc, but copy in the prior data.
    auto ret = NewMalloc(new_size);
    auto minimum_size = std::min(old_size, new_size);
    memcpy(ret, ptr, minimum_size);
    return ret;
}

void ScopedMPIRAllocators::NewFree(void* ptr, size_t size) {
    const auto alignedSize = (size + 63) & ~63;

    // Find offset relative to base of Allocated:
    const uint64_t offset = (uint8_t*)ptr - (uint8_t*)&Allocated[0][0];

    // Find index of Allocated:
    const size_t index = offset / BytesPerBlock;

    AllocationsAndFrees[index]--;
}

ScopedMPIRPrecision::ScopedMPIRPrecision(unsigned digits10) : saved_digits10(HighPrecision::thread_default_precision())
{
    HighPrecision::default_precision(digits10);
}
ScopedMPIRPrecision::~ScopedMPIRPrecision()
{
    HighPrecision::default_precision(saved_digits10);
}
void ScopedMPIRPrecision::reset(unsigned digits10)
{
    HighPrecision::default_precision(digits10);
}
void ScopedMPIRPrecision::reset()
{
    HighPrecision::default_precision(saved_digits10);
}

ScopedMPIRPrecisionOptions::ScopedMPIRPrecisionOptions(boost::multiprecision::variable_precision_options opts) : saved_options(HighPrecision::thread_default_variable_precision_options())
{
    HighPrecision::thread_default_variable_precision_options(opts);
}
ScopedMPIRPrecisionOptions::~ScopedMPIRPrecisionOptions()
{
    HighPrecision::thread_default_variable_precision_options(saved_options);
}
void ScopedMPIRPrecisionOptions::reset(boost::multiprecision::variable_precision_options opts)
{
    HighPrecision::thread_default_variable_precision_options(opts);
}
