#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>
#include <stddef.h>

#define HEAP_INIT_SIZE 0x10000
#define HEAP_MAX_SIZE 0xF0000
#define HEAP_MIN_SIZE 0x10000

#define MIN_ALLOC_SZ 4

#define MIN_WILDERNESS 0x2000
#define MAX_WILDERNESS 0x1000000

#define BIN_COUNT 31
#define BIN_MAX_IDX (BIN_COUNT - 1)

typedef struct node_t {
    uint64_t hole;
    uint64_t size;
    struct node_t* next;
    struct node_t* prev;
} node_t;

typedef struct { 
    node_t *header;
} footer_t;

typedef struct {
    node_t* head;
} bin_t;

typedef struct {
    uintptr_t start;
    uintptr_t end;
    bin_t binMemory[BIN_COUNT];
    bin_t *bins[BIN_COUNT];
} heap_t;

static uint64_t overhead = sizeof(footer_t) + sizeof(node_t);

#endif
