#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>
#include <stddef.h>

#define HEAP_INIT_SIZE 0x10000
#define MIN_ALLOC_SZ 16

#define MIN_WILDERNESS 0x2000
#define MAX_WILDERNESS 0x1000000

#define HEAP_BIN_COUNT 31
#define HEAP_BIN_MAX_IDX (HEAP_BIN_COUNT - 1)

typedef struct node_t {
    uint64_t hole;
    uint64_t size;

    static constexpr auto Magic = 0xDEADBEEFDEADBEEFllu;
    static constexpr auto ClearedMagic = 0x0llu;
    uint64_t magic;
    uint64_t checksum;

    static constexpr auto OffsetOfNext = sizeof(uint64_t) * 4;

    struct node_t* next;
    struct node_t* prev;
} node_t;

typedef struct { 
    node_t *header;
    uintptr_t pad;
} footer_t;

typedef struct {
    node_t* head;
    uintptr_t pad;
} bin_t;

typedef struct {
    uintptr_t start;
    uintptr_t end;
    bin_t binMemory[HEAP_BIN_COUNT];
    bin_t *bins[HEAP_BIN_COUNT];
} heap_t;

static uint64_t overhead = sizeof(footer_t) + sizeof(node_t);

#endif
