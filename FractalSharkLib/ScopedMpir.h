#pragma once

#include <mpir.h>
#include <vector>

// Forward declare GrowableVector
template <typename T>
class GrowableVector;

// Bump allocator that bumps downwards
class MPIRBoundedAllocator {
public:
    MPIRBoundedAllocator();
    ~MPIRBoundedAllocator();

    void InitScopedAllocators();

    static void InitTls();
    static void ShutdownTls();

private:
    void* (*ExistingMalloc) (size_t);
    void* (*ExistingRealloc) (void*, size_t, size_t);
    void (*ExistingFree) (void*, size_t);

    // It looks like we only need two blocks but we use four in case anything
    // changes in the RefOrbit calculation.  This all is pretty balanced together.
    static constexpr size_t NumBlocks = 4;
    static constexpr size_t BytesPerBlock = 256 * 1024;

    thread_local static uint8_t Allocated[NumBlocks][BytesPerBlock];
    thread_local static uint8_t* AllocatedEnd[NumBlocks];
    thread_local static size_t AllocatedIndex;
    thread_local static size_t AllocationsAndFrees[NumBlocks];

    // Currently, this one maxes out around 10.  So 10 live allocations at once.
    static std::atomic<size_t> MaxAllocatedDebug;

    static void* NewMalloc(size_t size);
    static void* NewRealloc(void* ptr, size_t old_size, size_t new_size);
    static void NewFree(void* ptr, size_t size);
};

class ThreadMemory {
public:
    ThreadMemory();
    ThreadMemory(const ThreadMemory &) = delete;
    ThreadMemory &operator=(const ThreadMemory &) = delete;
    ~ThreadMemory() = default;
    ThreadMemory(ThreadMemory &&) = default;
    ThreadMemory &operator=(ThreadMemory &&) = default;

    void* Allocate(size_t size);
    void Free(void* ptr, size_t size);

private:
    std::unique_ptr<GrowableVector<uint8_t>> m_Bump;
    std::vector<void*> m_FreedMemory;
    std::vector<size_t> m_FreedMemorySize;
};

// Bump allocator backed by a GrowableVector
class MPIRBumpAllocator {
public:
    MPIRBumpAllocator();
    ~MPIRBumpAllocator();

    void InitScopedAllocators();

    static void InitTls();
    static void ShutdownTls();
    static std::unique_ptr<ThreadMemory> GetAllocated(size_t index);
    static size_t GetAllocatorIndex();
    static bool IsBumpAllocatorInstalled();

private:
    void* (*ExistingMalloc) (size_t);
    void* (*ExistingRealloc) (void*, size_t, size_t);
    void (*ExistingFree) (void*, size_t);

    static std::atomic<size_t> MaxAllocatedDebug;
    static bool InstalledBumpAllocator;

    static void* NewMalloc(size_t size);
    static void* NewRealloc(void* ptr, size_t old_size, size_t new_size);
    static void NewFree(void* ptr, size_t size);

    static constexpr size_t NumAllocators = 5;
    static std::unique_ptr<ThreadMemory> m_Allocated[NumAllocators];
    static thread_local uint64_t m_ThreadIndex;
    static std::atomic<uint64_t> m_MaxIndex;
};

class MPIRPrecision
{
public:
    size_t m_SavedBits;

    MPIRPrecision(size_t bits);
    ~MPIRPrecision();
    void reset(size_t bits);
    void reset();
};
