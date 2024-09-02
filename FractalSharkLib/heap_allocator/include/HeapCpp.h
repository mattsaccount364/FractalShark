#pragma once

#include "heap.h"
#include "Vectors.h"

#include <stdexcept>

class HeapCpp {
public:
    HeapCpp(uintptr_t start) {
        init_heap(&heap_, start);
    }

    void *allocate(size_t size) {
        void *ptr = heap_alloc(&heap_, size);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(void *ptr) {
        if (ptr) {
            heap_free(&heap_, ptr);
        }
    }

    void expand(size_t size) {
        uint64_t result = ::expand(&heap_, size);
        if (!result) {
            throw std::runtime_error("HeapCpp expansion failed");
        }
    }

    void contract(size_t size) {
        ::contract(&heap_, size);
    }

    static uint64_t getBinIndex(size_t size) {
        return ::get_bin_index(size);
    }

    static void createFooter(node_t *head) {
        ::create_foot(head);
    }

    static footer_t *getFooter(node_t *head) {
        return ::get_foot(head);
    }

    static node_t *getWilderness(heap_t *heap) {
        return ::get_wilderness(heap);
    }

    // Destructor to handle any cleanup if necessary
    ~HeapCpp() {
        // Clean up if necessary
    }

private:
    heap_t heap_;
};
