#include "stdafx.h"
#include <cstdlib>

#include "include\HeapCpp.h" // Include your wrapper header
#include "..\Vectors.h"

#include "include\llist.h"

static int offset = sizeof(uintptr_t) * 2;
static constexpr auto PAGE_SIZE = 0x1000;

// sprintf etc use heap allocations in debug mode, so we can't use them here.
static constexpr bool LimitedDebugOut = !FractalSharkDebug;
static constexpr bool FullDebugOut = false;

//
// Overriding malloc/free is a bit tricky. The problem is that the C runtime library
// may call malloc/free internally, so if you override malloc/free, you may end up
// with a stack overflow. The solution is to use a global heap that is initialized
// before the C runtime library is initialized. This is done by defining a global
// instance of the HeapCpp class.
// 
// When the first attempt is made to allocate memory, the global heap is initialized.
// This initialization takes place during the C runtime library initialization, and is
// before any static constructors are called.  This has implications for the order of
// initialization of static objects and the "meaning" of static objects.
// 
// What we do here is call a constructor ourselves manually
//

// Define a global Heap instance
static HeapCpp globalHeap;


void VectorStaticInit();

void HeapCpp::InitGlobalHeap() {
    // Initialize the global heap
    if (globalHeap.Initialized) {
        return;
    }

    // Note: sprintf_s etc use heap allocations in debug mode, so we can't use them here.
    static_assert(sizeof(GrowableVector<uint8_t>) <= GrowableVectorSize);

    // GrowableVector<uint8_t>(AddPointOptions::EnableWithoutSave, L"HeapFile.bin");
    globalHeap.Growable = std::construct_at(
        reinterpret_cast<GrowableVector<uint8_t> *>(globalHeap.GrowableVectorMemory),
        AddPointOptions::EnableWithoutSave,
        L"HeapFile.bin");

    globalHeap.Growable->MutableResize(GrowByAmtBytes);
    globalHeap.Init();
}

HeapCpp::HeapCpp() {
    // It's very important that we don't do anything here, because this heap
    // may be used before its static constructor is called.
}

HeapCpp::~HeapCpp() {

    // So this is pretty excellent.  We don't want to touch anything here,
    // because this heap may still be used after it's destroyed.

    auto totalAllocs = CountAllocations();
    
    // Check if debugger attached.  Note this isn't necessarily the
    // end of the program, but it's a good place to check because one would
    // expect that it's close.
    if (IsDebuggerPresent()) {
        // Output totalAllocs to debug console in Visual Studio
        if constexpr (LimitedDebugOut) {
            char buffer[256];
            sprintf_s(buffer, "Total allocations remaining = %zu\n", totalAllocs);
            OutputDebugStringA(buffer);

            // Print Stats
            sprintf_s(buffer, "BytesAllocated = %zu\n", Stats.BytesAllocated);
            OutputDebugStringA(buffer);

            sprintf_s(buffer, "BytesFreed = %zu\n", Stats.BytesFreed);
            OutputDebugStringA(buffer);

            sprintf_s(buffer, "Allocations = %zu\n", Stats.Allocations);
            OutputDebugStringA(buffer);

            sprintf_s(buffer, "Frees = %zu\n", Stats.Frees);
            OutputDebugStringA(buffer);
        }
    }

    // Growable->Clear();
    // std::destroy_at(Growable);
    // Initialized = false;
}

// ========================================================
// this function initializes a new heap structure, provided
// an empty heap struct, and a place to start the heap
//
// NOTE: this function uses HEAP_INIT_SIZE to determine
// how large the heap is so make sure the same constant
// is used when allocating memory for your heap!
// ========================================================
void HeapCpp::Init() {
    // Initialize the heap
    for (size_t i = 0; i < BIN_COUNT; i++) {
        Heap.bins[i] = &Heap.binMemory[i];
    }

    node_t *init_region = reinterpret_cast<node_t *>(Growable->GetData());

    // first we create the initial region, this is the "wilderness" chunk
    // the heap starts as just one big chunk of allocatable memory
    init_region->hole = 1;
    init_region->size = (HEAP_INIT_SIZE)-sizeof(node_t) - sizeof(footer_t);

    CreateFooter(init_region); // create a foot (size must be defined)

    // now we add the region to the correct bin and setup the heap struct
    add_node(Heap.bins[GetBinIndex(init_region->size)], init_region);

    const uintptr_t start = reinterpret_cast<uintptr_t>(init_region);
    Heap.start = start;
    Heap.end = start + HEAP_INIT_SIZE;

    Initialized = true;
}

// ========================================================
// this is the allocation function of the heap, it takes
// the heap struct pointer and the size of the chunk we 
// want. this function will search through the bins until 
// it finds a suitable chunk. it will then split the chunk
// if neccesary and return the start of the chunk
// ========================================================

void *HeapCpp::Allocate(size_t size) {
    // We'll assume single-threaded initialization and that no races
    // occur here.  If you want to use this in a multi-threaded environment,
    // you'll need to add a mutex here.
    if (!Initialized) {
        VectorStaticInit();
        InitGlobalHeap();
    }

    // Round up to nearest 16 bytes
    size = (size + 0xF) & ~0xF;

    std::unique_lock<std::mutex> lock(Mutex);
    
    auto TryOnce = [&]() -> void * {
        // first get the bin index that this chunk size should be in
        auto index = GetBinIndex(size);
        // now use this bin to try and find a good fitting chunk!
        bin_t *temp = (bin_t *)Heap.bins[index];
        node_t *found = get_best_fit(temp, size);

        // while no chunk if found advance through the bins until we
        // find a chunk or get to the wilderness
        while (found == NULL) {
            if (index + 1 >= BIN_COUNT)
                return NULL;

            temp = Heap.bins[++index];
            found = get_best_fit(temp, size);
        }

        // if the differnce between the found chunk and the requested chunk
        // is bigger than the overhead (metadata size) + the min alloc size
        // then we should split this chunk, otherwise just return the chunk
        if ((found->size - size) > (overhead + MIN_ALLOC_SZ)) {
            // do the math to get where to split at, then set its metadata
            node_t *split = reinterpret_cast<node_t *>(((char *)found + overhead) + size);
            split->size = found->size - size - (overhead);
            split->hole = 1;

            CreateFooter(split); // create a footer for the split

            // now we need to get the new index for this split chunk
            // place it in the correct bin
            auto new_idx = GetBinIndex(split->size);
            add_node(Heap.bins[new_idx], split);

            found->size = size; // set the found chunks size
            CreateFooter(found); // since size changed, remake foot
        }

        found->hole = 0; // not a hole anymore
        remove_node(Heap.bins[index], found); // remove it from its bin

        // ==========================================

        // since we don't need the prev and next fields when the chunk
        // is in use by the user, we can clear these and return the
        // address of the next field
        found->prev = NULL;
        found->next = NULL;
        return &found->next;
    };

    auto *res = TryOnce();

    if (res == nullptr) {
        auto success = Expand(size * 2);
        if (success) {
            res = TryOnce();
        }
    }

    // these following lines are checks to determine if the heap should
    // be expanded or contracted
    // ==========================================
    node_t *wild = GetWilderness();
    if (wild->size < MIN_WILDERNESS) {
        auto success = Expand(PAGE_SIZE);
        if (success == false) {
            return NULL;
        }
    } else if (wild->size > MAX_WILDERNESS) {
        Contract(PAGE_SIZE);
    }

    if (reinterpret_cast<uintptr_t>(res) % 16 != 0) {
        throw std::exception();
    }

    Stats.Allocations++;
    Stats.BytesAllocated += size;
    return res;
}

// ========================================================
// this is the free function of the heap, it takes the 
// heap struct pointer and the pointer provided by the
// heap_alloc function. the given chunk will be possibly
// coalesced  and then placed in the correct bin
// ========================================================
void HeapCpp::Deallocate(void *ptr) {
    if (!ptr) {
        return;
    }

    std::unique_lock<std::mutex> lock(Mutex);
    assert(Initialized);
        
    bin_t *list;

    // the actual head of the node is not p, it is p minus the size
    // of the fields that precede "next" in the node structure
    // if the node being free is the start of the heap then there is
    // no need to coalesce so just put it in the right list
    node_t *head = (node_t *)((char *)ptr - offset);

    Stats.BytesFreed += head->size;
    Stats.Frees++;

    if (reinterpret_cast<uintptr_t>(head) == Heap.start) {
        head->hole = 1;
        add_node(Heap.bins[GetBinIndex(head->size)], head);
        return;
    }

    // these are the next and previous nodes in the heap, not the prev and next
    // in a bin. to find prev we just get subtract from the start of the head node
    // to get the footer of the previous node (which gives us the header pointer). 
    // to get the next node we simply get the footer and add the sizeof(footer_t).
    node_t *next = (node_t *)((char *)GetFooter(head) + sizeof(footer_t));
    node_t *prev = (node_t *)*((uintptr_t *)((char *)head - sizeof(footer_t)));

    // if the previous node is a hole we can coalese!
    if (prev->hole) {
        // remove the previous node from its bin
        list = Heap.bins[GetBinIndex(prev->size)];
        remove_node(list, prev);

        // re-calculate the size of thie node and recreate a footer
        prev->size += overhead + head->size;
        CreateFooter(prev);

        // previous is now the node we are working with, we head to prev
        // because the next if statement will coalesce with the next node
        // and we want that statement to work even when we coalesce with prev
        head = prev;
    }

    // if the next node is free coalesce!
    if (next->hole) {
        // remove it from its bin
        list = Heap.bins[GetBinIndex(next->size)];
        remove_node(list, next);

        // re-calculate the new size of head
        head->size += overhead + next->size;

        // clear out the old metadata from next
        footer_t *old_foot = GetFooter(next);
        old_foot->header = 0;
        next->size = 0;
        next->hole = 0;

        // make the new footer!
        CreateFooter(head);
    }

    // this chunk is now a hole, so put it in the right bin!
    head->hole = 1;
    add_node(Heap.bins[GetBinIndex(head->size)], head);
}

bool HeapCpp::Expand(size_t deltaSizeBytes) {
    // Mutex must be held

    // Grow by a minimum of a megabyte
    // Ensure the size to expand is aligned to page size (0x1000).
    const auto growSizeBytes = std::max(size_t(GrowByAmtBytes), deltaSizeBytes * 2);

    // Round up to the nearest page size.
    const auto size = (growSizeBytes + 0xFFF) & ~0xFFF;

    // Calculate the new end of the heap.
    void *new_end = (char *)Heap.end + size;

    // Now we need to grow the GrowableVector to match the new heap size, if needed.
    // That'll ensure the memory is writable and we can use it.
    Growable->GrowVectorByAmount(size);

    // Check if the new end is within the maximum heap size.

    // Get the current wilderness block (last block in the heap).
    node_t *wild = GetWilderness();

    // If the wilderness block is a hole (free), coalesce it with the new free block.
    if (wild->hole) {
        // Remove the wilderness block from its current bin.
        auto wild_bin_index = GetBinIndex(wild->size);
        remove_node(Heap.bins[wild_bin_index], wild);

        // Increase the size of the wilderness block by adding the expanded memory.
        wild->size += size;

        // Update the footer of the coalesced wilderness block.
        CreateFooter(wild);

        // Add the coalesced wilderness block back into the correct bin.
        auto new_bin_index = GetBinIndex(wild->size);
        add_node(Heap.bins[new_bin_index], wild);
    } else {
        // If the wilderness block is not free, create a new free block at the old heap end.
        node_t *new_free_block = (node_t *)Heap.end;
        new_free_block->size = size - overhead; // New block size, minus overhead (header and footer).
        new_free_block->hole = 1; // Mark it as a free block (hole).

        // Create the footer for the new free block.
        CreateFooter(new_free_block);

        // Add the new free block to the appropriate bin.
        auto new_bin_index = GetBinIndex(new_free_block->size);
        add_node(Heap.bins[new_bin_index], new_free_block);
    }

    // Update the heap's end pointer to the new expanded end.
    Heap.end = reinterpret_cast<uintptr_t>(new_end);

    return true;
}

bool HeapCpp::Contract(size_t size) {
    // Mutex must be held
    
    // Ensure the size to contract is aligned to page size (0x1000).
    size = (size + 0xFFF) & ~0xFFF;

    node_t *wild = GetWilderness();

    // Only contract if the wilderness region is large enough.
    if (wild->size < size) {
        return false;
    }

    // Shrink the wilderness region by reducing its size.
    wild->size -= size;
    CreateFooter(wild);

    // Update heap's end pointer to reflect the contraction.
    auto newEnd = Heap.end - size;
    Heap.end = newEnd;

    // Simulate returning the memory to the system, in real systems, this would involve OS calls.

    return true;
}

size_t HeapCpp::CountAllocations() const {
    size_t count = 0;

    // Start at the beginning of the heap
    uintptr_t current_address = Heap.start;

    // Traverse through the heap until we reach the end
    while (current_address < Heap.end) {
        // Get the current node (header) at the current address
        node_t *current_node = reinterpret_cast<node_t *>(current_address);

        // If it's not a hole, it means it's an active allocation
        if (current_node->hole == 0) {
            count++;
        }

        // Move to the next node by advancing by the size of the current node plus the overhead
        current_address += sizeof(node_t) + current_node->size + sizeof(footer_t);
    }

    return count;
}

// ========================================================
// this function is the hashing function that converts
// size => bin index. changing this function will change 
// the binning policy of the heap. right now it just 
// places any allocation < 8 in bin 0 and then for anything
// above 8 it bins using the log base 2 of the size
// ========================================================
uint64_t HeapCpp::GetBinIndex(size_t size) {
    int index = 0;
    constexpr auto minAlignment = 16;
    size = size < minAlignment ? minAlignment : size;

    while (size >>= 1) index++;
    index -= 4;

    if (index > BIN_MAX_IDX) {
        assert(false);
        index = BIN_MAX_IDX;
    }
    return index;
}

// ========================================================
// this function will create a footer given a node
// the node's size must be set to the correct value!
// ========================================================
void HeapCpp::CreateFooter(node_t *head) {
    footer_t *foot = GetFooter(head);
    foot->header = head;

}

// ========================================================
// this function will get the footer pointer given a node
// ========================================================
footer_t *HeapCpp::GetFooter(node_t *head) {
    return (footer_t *)((char *)head + sizeof(node_t) + head->size);
}

// ========================================================
// this function will get the wilderness node given a 
// heap struct pointer
//
// NOTE: this function banks on the heap's end field being
// correct, it simply uses the footer at the end of the 
// heap because that is always the wilderness
// ========================================================
node_t *HeapCpp::GetWilderness() {
    footer_t *wild_foot = (footer_t *)((char *)Heap.end - sizeof(footer_t));
    return wild_foot->header;
}

//////////////////////////////////////////////////////////////////////////

void *CppMalloc(size_t size) {
    auto *res = globalHeap.Allocate(size);

    // Output res in hex to debug console in Visual Studio
    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "malloc(%zu) = %p\n", size, res);
        OutputDebugStringA(buffer);
    }

    return res;
}

void CppFree(void *ptr) {
    // Output ptr in hex to debug console in Visual Studio
    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "free(%p)\n", ptr);
        OutputDebugStringA(buffer);
    }

    globalHeap.Deallocate(ptr);
}

void *CppRealloc(void *ptr, size_t newSize) {
    if (!ptr) {
        return CppMalloc(newSize);
    }

    if (newSize == 0) {
        CppFree(ptr);
        return nullptr;
    }

    void *newPtr = CppMalloc(newSize);
    if (!newPtr) {
        return nullptr;
    }

    // Find the old size:
    const auto node = reinterpret_cast<node_t *>(
        reinterpret_cast<uintptr_t>(ptr) - offset);

    // Assume the size is stored somehow, or manage separately in the Heap class
    std::memcpy(newPtr, ptr, std::min(node->size, newSize));
    CppFree(ptr);
    return newPtr;
}

// Override global malloc, free, and realloc functions
extern "C" {
    __declspec(restrict)
    void *malloc(size_t size) {
        return CppMalloc(size);
    }

    // We rely on linking with /force:multiple to avoid LNK2005
#ifdef _DEBUG
    void *_malloc_dbg(
        size_t size,
        int /*blockType*/,
        const char * /*filename*/,
        int /*line*/) {

        return CppMalloc(size);
    }
#endif

    __declspec(restrict)
    void *calloc(size_t num, size_t size) {
        auto *res = CppMalloc(num * size);
        if (res) {
            std::memset(res, 0, num * size);
        }
        return res;
    }

#ifdef _DEBUG
    // We rely on linking with /force:multiple to avoid LNK2005
    void *_calloc_dbg(
        size_t num,
        size_t size,
        int /*blockType*/,
        const char * /*filename*/,
        int /*line*/) {

        return calloc(num, size);
    }
#endif

    __declspec(restrict)
    void *aligned_alloc(size_t alignment, size_t size) {
        auto res = CppMalloc(size);
        assert(reinterpret_cast<uintptr_t>(res) % alignment == 0);
        return res;
    }

    void free(void *ptr) {
        CppFree(ptr);
    }

#ifdef _DEBUG
    void _free_dbg(void *ptr, int /*blockType*/) {
        CppFree(ptr);
    }
#endif

    __declspec(restrict)
    void *realloc(void *ptr, size_t newSize) {
        return CppRealloc(ptr, newSize);
    }

    char *strdup(const char *s) {
        size_t len = strlen(s) + 1;
        char *d = (char *)malloc(len);
        if (d) {
            memcpy(d, s, len);
        }
        return d;
    }

    __declspec(restrict)
    char *strndup(const char *s, size_t n) {
        size_t len = strnlen(s, n);
        char *d = (char *)malloc(len + 1);
        if (d) {
            memcpy(d, s, len);
            d[len] = '\0';
        }
        return d;
    }

    __declspec(restrict)
    char *realpath(const char *fname, char *resolved_name) {
        return _fullpath(resolved_name, fname, _MAX_PATH);
    }
} // extern "C"

void   operator delete(void *ptr) {
    free(ptr);
}

void operator delete[](void *ptr) {
    free(ptr);
}

void *operator new(size_t size) noexcept(false) {
    auto *res = malloc(size);
    if (!res) {
        throw std::bad_alloc();
    }
    return res;
}

// Array form of new:
void *operator new[](size_t size) {
    return operator new(size);
}

void *operator new(std::size_t n, std::align_val_t align) noexcept(false) {
    return aligned_alloc(static_cast<size_t>(align), n);
}

void *operator new[](std::size_t n, std::align_val_t align) noexcept(false) {
    return aligned_alloc(static_cast<size_t>(align), n);
}

void *operator new (std::size_t count, const std::nothrow_t &/*tag*/) noexcept {
    return malloc(count);
}

void *operator new[](std::size_t count, const std::nothrow_t &/*tag*/) noexcept {
    return malloc(count);
}

void *operator new  (std::size_t count, std::align_val_t al, const std::nothrow_t &) noexcept {
    return aligned_alloc(static_cast<size_t>(al), count);
}

void *operator new[](std::size_t count, std::align_val_t al, const std::nothrow_t &) noexcept {
    return aligned_alloc(static_cast<size_t>(al), count);
}
