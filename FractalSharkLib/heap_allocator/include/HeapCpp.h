#pragma once

#include "heap.h"

#include <stdexcept>
#include <mutex>


template<typename T>
class GrowableVector;

class HeapCpp {
public:
    HeapCpp();
    ~HeapCpp();

    static void InitGlobalHeap();

    // No ShutdownGlobalHeap().  We don't know when to call it.
    // There's no guarantee when the "last" allocation will be made.

    void Init();

    void *Allocate(size_t size);

    void Deallocate(void *ptr);

    bool Expand(size_t deltaSizeBytes);
    bool Contract(size_t deltaSizeBytes);

    size_t CountAllocations() const;

private:
    uint64_t GetBinIndex(size_t size);
    void CreateFooter(node_t *head);
    footer_t *GetFooter(node_t * head);
    node_t *GetWilderness();

    bool Initialized;
    heap_t Heap;
    std::mutex Mutex;

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
