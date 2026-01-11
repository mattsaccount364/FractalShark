#pragma once

#include "heap.h"

#include <stdexcept>
#include <mutex>

//
// This implementation is a modified version of the simple one found here
// https://github.com/CCareaga/heap_allocator
//

template<typename T>
class GrowableVector;

void RegisterHeapCleanup();


class HeapCpp {

public:
    // This is a singleton.  The global heap is the only one that should be used.
    // The constructor does nothing, which is important because the singleton
    // may be created before the static constructor runs.
    static void InitGlobalHeap();
    
    HeapCpp();

    // No ShutdownGlobalHeap().  We don't know when to call it.
    // There's no guarantee when the "last" allocation will be made.
    // The destructor is "close" but not guaranteed to be at the "end."
    ~HeapCpp();

    // Initialize the heap.  This must be called before any other methods.
    void Init();

    // Allocate and deallocate memory.  Returns nullptr on failure.
    void *Allocate(size_t size);
    void Deallocate(void *ptr);

    // Expand and contract the heap.  Returns false on failure.
    // Also expands the underlying GrowableVector.  Does not contract it.
    bool Expand(size_t deltaSizeBytes);
    bool Contract(size_t deltaSizeBytes);

    // Count the number of outstanding allocations.
    // This is useful for detecting memory leaks.
    size_t CountAllocations() const;

private:
    // Returns the bin index for a given allocation size.
    uint64_t GetBinIndex(size_t size);

    void CreateFooter(node_t *head);

    footer_t *GetFooter(node_t * head);

    node_t *GetWilderness();

    bool Initialized;
    heap_t Heap;

    // Wow:
    std::mutex *Mutex;
    char MutexBuffer[256];

    struct StatsCollection {
        size_t BytesAllocated;
        size_t BytesFreed;
        size_t Allocations;
        size_t Frees;
    };

    StatsCollection Stats;

    static constexpr auto GrowByAmtBytes = 1024 * 1024; // 1MB
    static constexpr auto GrowableVectorSize = 2048;
    uint8_t GrowableVectorMemory[GrowableVectorSize];
    GrowableVector<uint8_t> *Growable;
};

HeapCpp &GlobalHeap();
