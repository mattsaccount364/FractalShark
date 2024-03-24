#include "stdafx.h"

#include "ScopedMpir.h"
#include "HighPrecision.h"
#include "Vectors.h"

thread_local uint8_t MPIRBoundedAllocator::Allocated[NumBlocks][BytesPerBlock];
thread_local uint8_t *MPIRBoundedAllocator::AllocatedEnd[NumBlocks];
thread_local size_t MPIRBoundedAllocator::AllocatedIndex = 0;
thread_local size_t MPIRBoundedAllocator::AllocationsAndFrees[NumBlocks];
std::atomic<size_t> MPIRBoundedAllocator::MaxAllocatedDebug = 0;
bool MPIRBumpAllocator::InstalledBumpAllocator = false;

static constexpr bool DebugInstrument = false;

MPIRBoundedAllocator::MPIRBoundedAllocator() :
    ExistingMalloc{},
    ExistingRealloc{},
    ExistingFree{} {
}

MPIRBoundedAllocator::~MPIRBoundedAllocator() {
    if (ExistingMalloc == nullptr) {
        return;
    }

    mp_set_memory_functions(
        ExistingMalloc,
        ExistingRealloc,
        ExistingFree);
}

bool MPIRBumpAllocator::IsBumpAllocatorInstalled() {
    return InstalledBumpAllocator;
}

void MPIRBoundedAllocator::InitScopedAllocators() {
    mp_get_memory_functions(
        &ExistingMalloc,
        &ExistingRealloc,
        &ExistingFree);

    mp_set_memory_functions(
        NewMalloc,
        NewRealloc,
        NewFree);
}

void MPIRBoundedAllocator::InitTls() {
    memset(Allocated, 0, sizeof(Allocated));
    memset(AllocationsAndFrees, 0, sizeof(AllocationsAndFrees));
    AllocatedIndex = 0;

    for (size_t i = 0; i < NumBlocks; i++) {
        AllocatedEnd[i] = &Allocated[i][BytesPerBlock];
    }
}

void MPIRBoundedAllocator::ShutdownTls() {
    // If there are any allocations left, we have a memory leak.
    if constexpr (DebugInstrument) {
        for (size_t i = 0; i < NumBlocks; i++) {
            if (AllocationsAndFrees[i] != 0) {
                DebugBreak();
            }
        }
    }
}

void* MPIRBoundedAllocator::NewMalloc(size_t size) {
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

void* MPIRBoundedAllocator::NewRealloc(void* /*ptr*/, size_t /*old_size*/, size_t /*new_size*/) {
    // This one is like new_malloc, but copy in the prior data.
    //auto ret = NewMalloc(new_size);
    //auto minimum_size = std::min(old_size, new_size);
    //memcpy(ret, ptr, minimum_size);

    if constexpr (DebugInstrument) {
        DebugBreak();
    }
    return nullptr;
}

void MPIRBoundedAllocator::NewFree(void* ptr, size_t /*size*/) {
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

//////////////////////////////////////////////////////////////////////////

std::atomic<size_t> MPIRBumpAllocator::MaxAllocatedDebug = 0;
std::unique_ptr<GrowableVector<uint8_t>> MPIRBumpAllocator::m_Allocated[NumAllocators];
thread_local uint64_t MPIRBumpAllocator::m_ThreadIndex;
std::atomic<uint64_t> MPIRBumpAllocator::m_MaxIndex = 0;

MPIRBumpAllocator::MPIRBumpAllocator() :
    ExistingMalloc{},
    ExistingRealloc{},
    ExistingFree{} {
}

MPIRBumpAllocator::~MPIRBumpAllocator() {
    if (ExistingMalloc == nullptr) {
        return;
    }

    mp_set_memory_functions(
        ExistingMalloc,
        ExistingRealloc,
        ExistingFree);

    InstalledBumpAllocator = false;
}

void MPIRBumpAllocator::InitScopedAllocators() {
    mp_get_memory_functions(
        &ExistingMalloc,
        &ExistingRealloc,
        &ExistingFree);

    mp_set_memory_functions(
        NewMalloc,
        NewRealloc,
        NewFree);

    InstalledBumpAllocator = true;

    // Note: initialize m_MaxIndex to 1 so that the first thread to call InitTls will get index 1.
    // That way, if a thread doesn't call InitTls, it will get index 0, and it will AV
    // if it tries to use the allocator. This is a good thing, because it means that
    // we will catch the error early, rather than having it silently corrupt memory.
    m_MaxIndex = 1;
    m_ThreadIndex = 0;

    // Initialize all the allocators except at index 0.
    // Index 0 is left uninitialized so that we can catch errors if a thread doesn't call InitTls.
    m_Allocated[0] = nullptr;
    for (size_t i = 1; i < NumAllocators; i++) {
        m_Allocated[i] = std::make_unique<GrowableVector<uint8_t>>(AddPointOptions::EnableWithoutSave, L"");
    }
}

void MPIRBumpAllocator::InitTls() {
    // Atomically increment m_MaxIndex and use the result as the index for this thread.
    // Returns the previous value of m_MaxIndex.
    m_ThreadIndex = m_MaxIndex.fetch_add(1, std::memory_order_seq_cst);
}

void MPIRBumpAllocator::ShutdownTls() {
    if constexpr (DebugInstrument) {
        if (m_Allocated[m_ThreadIndex]->GetSize() != 0) {
            DebugBreak();
        }
    }

    m_Allocated[m_ThreadIndex].reset();
}

std::unique_ptr<GrowableVector<uint8_t>> MPIRBumpAllocator::GetAllocated() {
    return std::move(m_Allocated[m_ThreadIndex]);
}

void* MPIRBumpAllocator::NewMalloc(size_t size) {
    // Round up size to nearest 64-byte multiple
    size = (size + 63) & ~63;

    auto curAllocator = m_Allocated[m_ThreadIndex].get();
    curAllocator->GrowVectorIfNeeded();
    curAllocator->MutableResize(curAllocator->GetSize() + size);
    return curAllocator->GetData() + curAllocator->GetSize() - size;

    //return malloc(size);
}

void* MPIRBumpAllocator::NewRealloc(void* ptr, size_t old_size, size_t new_size) {
    // Implement realloc by copying memory to new location and freeing old location.
    auto minSize = std::min(old_size, new_size);
    void *newLoc = NewMalloc(new_size);
    memcpy(newLoc, ptr, minSize);
    return newLoc;

    //return realloc(ptr, new_size);
}

void MPIRBumpAllocator::NewFree(void* /*ptr*/, size_t /*size*/) {    
    // Do nothing - leak it!
}

MPIRPrecision::MPIRPrecision(size_t bits) : m_SavedBits(HighPrecision::defaultPrecisionInBits())
{
    HighPrecision::defaultPrecisionInBits(bits);
}

MPIRPrecision::~MPIRPrecision()
{
    HighPrecision::defaultPrecisionInBits(m_SavedBits);
}

void MPIRPrecision::reset(size_t bits)
{
    HighPrecision::defaultPrecisionInBits(bits);
}

void MPIRPrecision::reset()
{
    HighPrecision::defaultPrecisionInBits(m_SavedBits);
}
