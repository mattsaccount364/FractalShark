#include "stdafx.h"

#include "ScopedMpir.h"
#include "HighPrecision.h"

thread_local uint8_t ScopedMPIRAllocators::Allocated[NumBlocks][BytesPerBlock];
thread_local uint8_t *ScopedMPIRAllocators::AllocatedEnd[NumBlocks];
thread_local size_t ScopedMPIRAllocators::AllocatedIndex = 0;
thread_local size_t ScopedMPIRAllocators::AllocationsAndFrees[NumBlocks];
std::atomic<size_t> ScopedMPIRAllocators::MaxAllocatedDebug = 0;

static constexpr bool DebugInstrument = false;

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
    memset(AllocationsAndFrees, 0, sizeof(AllocationsAndFrees));
    AllocatedIndex = 0;

    for (size_t i = 0; i < NumBlocks; i++) {
        AllocatedEnd[i] = &Allocated[i][BytesPerBlock];
    }
}

void ScopedMPIRAllocators::ShutdownTls() {
    // If there are any allocations left, we have a memory leak.
    if constexpr (DebugInstrument) {
        for (size_t i = 0; i < NumBlocks; i++) {
            if (AllocationsAndFrees[i] != 0) {
                DebugBreak();
            }
        }
    }
}

void* ScopedMPIRAllocators::NewMalloc(size_t size) {
    uint8_t *newPtr = AllocatedEnd[AllocatedIndex] - size;

    // Round down to requested alignment:
    newPtr = (uint8_t*)((uintptr_t)newPtr & ~63);

    if (newPtr >= &Allocated[AllocatedIndex][0]) {
        AllocationsAndFrees[AllocatedIndex]++;
        AllocatedEnd[AllocatedIndex] = newPtr;

        if constexpr (DebugInstrument) {
            // This all is a bit overkill but at least it's guaranteed to work.
            // Keeps track of maximum allocations required.
            for (;;) {
                auto oldVal = MaxAllocatedDebug.load(std::memory_order_seq_cst);
                if (AllocationsAndFrees[AllocatedIndex] > oldVal) {
                    auto result = MaxAllocatedDebug.compare_exchange_weak(
                        oldVal,
                        AllocationsAndFrees[AllocatedIndex],
                        std::memory_order_seq_cst);

                    if (result) {
                        break;
                    }
                }
                else {
                    break;
                }
            }
        }

        return newPtr;
    } else {
        // Find the next block with no allocations:
        size_t InitIndex = AllocatedIndex;
        for (;;) {
            AllocatedIndex = (AllocatedIndex + 1) % NumBlocks;
            if (AllocationsAndFrees[AllocatedIndex] == 0) {
                newPtr = &Allocated[AllocatedIndex][BytesPerBlock] - size;
                newPtr = (uint8_t*)((uintptr_t)newPtr & ~63);
                AllocationsAndFrees[AllocatedIndex]++;
                AllocatedEnd[AllocatedIndex] = newPtr;
                return newPtr;
            }

            if constexpr (DebugInstrument) {
                if (AllocatedIndex == InitIndex) {
                    DebugBreak();
                }
            }
        }
    }
}

void* ScopedMPIRAllocators::NewRealloc(void* /*ptr*/, size_t /*old_size*/, size_t /*new_size*/) {
    // This one is like new_malloc, but copy in the prior data.
    //auto ret = NewMalloc(new_size);
    //auto minimum_size = std::min(old_size, new_size);
    //memcpy(ret, ptr, minimum_size);

    if constexpr (DebugInstrument) {
        DebugBreak();
    }
    return nullptr;
}

void ScopedMPIRAllocators::NewFree(void* ptr, size_t /*size*/) {
    // Find offset relative to base of Allocated:
    const uint64_t offset = (uint8_t*)ptr - (uint8_t*)&Allocated[0][0];

    // Find index of Allocated:
    const size_t index = offset / BytesPerBlock;

    if constexpr (DebugInstrument) {
        if (index >= NumBlocks) {
            DebugBreak();
        }

        if (AllocationsAndFrees[index] == 0) {
            DebugBreak();
        }
    }

    --AllocationsAndFrees[index];
}

ScopedMPIRPrecision::ScopedMPIRPrecision(size_t bits) : m_SavedBits(HighPrecision::defaultPrecisionInBits())
{
    HighPrecision::defaultPrecisionInBits(bits);
}

ScopedMPIRPrecision::~ScopedMPIRPrecision()
{
    HighPrecision::defaultPrecisionInBits(m_SavedBits);
}

void ScopedMPIRPrecision::reset(size_t bits)
{
    HighPrecision::defaultPrecisionInBits(bits);
}

void ScopedMPIRPrecision::reset()
{
    HighPrecision::defaultPrecisionInBits(m_SavedBits);
}
