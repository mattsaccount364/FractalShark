#pragma once

#include "heap.h"

#include <stdexcept>
#include <mutex>

void InitGlobalHeap();
void ShutdownGlobalHeap();

class HeapCpp {
public:
    HeapCpp();
    ~HeapCpp();

    void Init();

    void *Allocate(size_t size);

    void Deallocate(void *ptr);

    bool Expand(size_t deltaSizeBytes);
    bool Contract(size_t deltaSizeBytes);

private:
    uint64_t GetBinIndex(size_t size);
    void CreateFooter(node_t *head);
    footer_t *GetFooter(node_t * head);
    node_t *GetWilderness();

    bool Initialized;
    heap_t Heap;
    std::mutex Mutex;
};
