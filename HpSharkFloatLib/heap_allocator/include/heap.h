#ifndef HEAP_H
#define HEAP_H

#include <stddef.h>
#include <stdint.h>

#define HEAP_INIT_SIZE 0x10000
#define MIN_ALLOC_SZ 16

#define MIN_WILDERNESS 0x2000
#define MAX_WILDERNESS 0x1000000

#define HEAP_BIN_COUNT 31
#define HEAP_BIN_MAX_IDX (HEAP_BIN_COUNT - 1)

typedef struct node_t {

    // Put the node into a known "free and not linked" state.
    // Caller supplies the payload size (not including header/footer).
    void
    init_free_node_unlinked(uint64_t sz)
    {
        hole = 1;
        user_size = sz;
        actual_size = sz;

        in_bin = NotInBin;
        magic = ClearedMagic;

        // Leave in_bin and in_bin_gen alone.
        // Those are changed via mark_in_bin and mark_not_in_bin.
        // Now treated as freelist links (allocator-owned).
        next = nullptr;
        prev = nullptr;
    }

    uint64_t hole;
    uint64_t user_size;
    uint64_t actual_size;

    static constexpr uint64_t Magic = 0xDEADBEEFDEADBEEFllu;
    static constexpr uint64_t ClearedMagic = 0xABCDABCDABCDABCDllu;
    static constexpr uint64_t NotInBin = 0xFFFFFFFFFFFFFFFFull;

    uint64_t magic;
    uint64_t in_bin;
    uint64_t in_bin_gen;

    struct node_t *next;
    struct node_t *prev;

} node_t;

typedef struct {
    node_t *header;
    uintptr_t pad;
} footer_t;

typedef struct {
    node_t *head;
    uintptr_t pad;
} bin_t;

typedef struct {
    uintptr_t start;
    uintptr_t end;
    bin_t binMemory[HEAP_BIN_COUNT];
    bin_t *bins[HEAP_BIN_COUNT];
} heap_t;

static constexpr uint64_t overhead = sizeof(footer_t) + sizeof(node_t);

#endif
