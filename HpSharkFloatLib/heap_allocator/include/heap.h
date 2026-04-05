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
        checksum = 0;
        poisoned = NotPoisoned;
        // Note: alloc_gen is NOT reset here. It tracks whether the block
        // was ever handed to user code (set in Allocate, verified on Deallocate).

        // Leave in_bin and in_bin_gen alone.
        // Those are changed via mark_in_bin and mark_not_in_bin.
        // Now treated as freelist links (allocator-owned).
        next = nullptr;
        prev = nullptr;
    }

    // Lightweight XOR checksum over metadata fields.
    // Detects wild-pointer scribbles that corrupt node headers.
    uint64_t
    ComputeChecksum() const
    {
        return hole ^ user_size ^ actual_size ^ magic ^ in_bin ^ in_bin_gen;
    }

    static constexpr uint64_t WasPoisoned = 0xCD00CD00CD00CD00ull;
    static constexpr uint64_t NotPoisoned = 0;

    uint64_t hole;
    uint64_t user_size;
    uint64_t actual_size;

    static constexpr uint64_t Magic = 0xDEADBEEFDEADBEEFllu;
    static constexpr uint64_t ClearedMagic = 0xABCDABCDABCDABCDllu;
    static constexpr uint64_t NotInBin = 0xFFFFFFFFFFFFFFFFull;
    static constexpr uint64_t HeadGuard = 0xFEEDBEEFFEEDBEEFull;

    uint64_t magic;
    uint64_t in_bin;
    uint64_t in_bin_gen;
    uint64_t checksum;
    uint64_t alloc_gen;
    uint64_t poisoned;  // WasPoisoned after free, NotPoisoned otherwise

    struct node_t *next;
    struct node_t *prev;

    // Head guard: last field before user payload.
    // A backward scribble from user data hits this first.
    uint64_t head_guard;

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
