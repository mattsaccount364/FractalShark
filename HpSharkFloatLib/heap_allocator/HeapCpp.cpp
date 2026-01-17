#include "include\HeapCpp.h" // Include your wrapper header
#include "..\DbgHeap.h"
#include "..\EarlyCommandLine.h"
#include "include\HeapPanic.h"
#include "include\llist.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <type_traits>
#include <cstring>
#include <new>
#include <errno.h>

#include "..\Vectors.h"

#define NOMINMAX
#include <windows.h>

#pragma optimize("", off)


FancyHeap EnableFractalSharkHeap;

static constexpr auto PAGE_SIZE = 0x1000;

// sprintf etc use heap allocations in debug mode, so we can't use them here.
static constexpr bool LimitedDebugOut = !FractalSharkDebug;
static constexpr bool FullDebugOut = false;

// --------------------------------------------------------
// System heap (Win32 process heap) forwarding
// --------------------------------------------------------

// In 'safemode' or when the fancy heap is disabled, we bypass the CRT allocator
// completely and use the Win32 process heap. This avoids GetProcAddress and
// works safely during very early initialization (TLS/CRT init).

static HeapCpp gHeap;
static HANDLE g_proc_heap = nullptr;

static __forceinline HANDLE
ProcHeap()
{
    if (!g_proc_heap) {
        g_proc_heap = GetProcessHeap();
    }
    return g_proc_heap;
}

static __forceinline void *
SysMalloc(size_t n)
{
    if (n == 0) {
        n = 1;
    }
    return HeapAlloc(ProcHeap(), 0, n);
}

static __forceinline void *
SysCalloc(size_t n, size_t sz)
{
    if (sz != 0 && n > (SIZE_MAX / sz)) {
        return nullptr;
    }
    size_t total = n * sz;
    if (total == 0) {
        total = 1;
    }
    return HeapAlloc(ProcHeap(), HEAP_ZERO_MEMORY, total);
}

static __forceinline void
SysFree(void *p)
{
    if (!p) {
        return;
    }
    HeapFree(ProcHeap(), 0, p);
}

static __forceinline void *
SysRealloc(void *p, size_t n)
{
    if (n == 0) {
        SysFree(p);
        return nullptr;
    }
    if (!p) {
        return SysMalloc(n);
    }
    return HeapReAlloc(ProcHeap(), 0, p, n);
}

static __forceinline bool
IsPow2(size_t x)
{
    return x && ((x & (x - 1)) == 0);
}

void
CleanupHeap()
{
    gHeap.~HeapCpp();
}

void
RegisterHeapCleanup()
{
    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        atexit(CleanupHeap);
    } else if (EnableFractalSharkHeap == FancyHeap::Disable) {
        // Heh, so on this path we bypass the fancy heap but we still must init this
        // logic or the growable vector fails in other contexts.
        VectorStaticInit();
    } else {
        HeapPanic("FractalShark heap disabled via command line");
    }
}

void VectorStaticInit();

HeapCpp &
GlobalHeap()
{
    return gHeap;
}

void
HeapCpp::InitGlobalHeap()
{
    if (EnableFractalSharkHeap == FancyHeap::Disable) {
        return;
    }

    // Construct HeapCpp in raw storage (no dynamic initialization at static-init time;
    // construction happens here under your control).
    auto &globalHeap = GlobalHeap();

    if (globalHeap.Initialized) {
        return;
    }

    // Note: sprintf_s etc use heap allocations in debug mode, so we can't use them here.
    static_assert(sizeof(GrowableVector<uint8_t>) <= GrowableVectorSize);

    // GrowableVector<uint8_t>(AddPointOptions::EnableWithoutSave, L"HeapFile.bin");
    globalHeap.Growable =
        std::construct_at(reinterpret_cast<GrowableVector<uint8_t> *>(globalHeap.GrowableVectorMemory),
                          AddPointOptions::EnableWithoutSave,
                          L"HeapFile.bin");

    globalHeap.Growable->MutableResize(GrowByAmtBytes);
    globalHeap.Init();

    // TODO
    // RegisterHeapCleanup();
}

HeapCpp::HeapCpp()
{
    if (IsDebuggerPresent()) {
        char buffer[256];
        sprintf_s(buffer, "HeapCpp Constructor called\n");
        OutputDebugStringA(buffer);
    }

    // It's very important that we don't do anything here, because this heap
    // may be used before its static constructor is called.
}

HeapCpp::~HeapCpp()
{
    Mutex->~mutex();

    // So this is pretty excellent.  We don't want to touch anything here,
    // because this heap may still be used after it's destroyed.

    auto totalAllocs = CountAllocations();

    // Check if debugger attached.  Note this isn't necessarily the
    // end of the program, but it's a good place to check because one would
    // expect that it's close.
    if (IsDebuggerPresent()) {
        // Output totalAllocs to debug console in Visual Studio
        char buffer[256];
        sprintf_s(buffer, "Total allocations remaining = %zu\n", totalAllocs);
        OutputDebugStringA(buffer);

        // Print Stats
        sprintf_s(buffer, "BytesAllocated = %zu\n", Stats.BytesAllocated);
        OutputDebugStringA(buffer);

        sprintf_s(buffer, "BytesFreed = %zu\n", Stats.BytesFreed);
        OutputDebugStringA(buffer);

        sprintf_s(buffer, "Delta bytes allocated = %zu\n", Stats.BytesAllocated - Stats.BytesFreed);
        OutputDebugStringA(buffer);

        sprintf_s(buffer, "Allocations = %zu\n", Stats.Allocations);
        OutputDebugStringA(buffer);

        sprintf_s(buffer, "Frees = %zu\n", Stats.Frees);
        OutputDebugStringA(buffer);

        sprintf_s(buffer, "Delta allocations = %zu\n", Stats.Allocations - Stats.Frees);
        OutputDebugStringA(buffer);
    }

    // Growable->Clear();
    // std::destroy_at(Growable);
    // Initialized = false;
}

static inline void
assert_free_unlinked_for_resize(const node_t *n, const char *where)
{
    if (n->hole) {
        HEAP_ASSERT(n->in_bin == node_t::NotInBin, where);
        HEAP_ASSERT(n->next == nullptr && n->prev == nullptr, where);
    }
}

static inline void
assert_linked_in(node_t *n, uint64_t idx, const char *msg)
{
    HEAP_ASSERT(n->in_bin == idx, msg);
}

void
SetMagicAtEnd(void *newBuffer, size_t bufferSize)
{
    // Set magic at end of buffer to detect overruns
    uint64_t *magicPtr =
        reinterpret_cast<uint64_t *>((char *)newBuffer + bufferSize - 2 * sizeof(uint64_t));
    magicPtr[0] = node_t::Magic;
    magicPtr[1] = node_t::Magic;
}
void
VerifyAndClearMagicAtEnd(void *buffer, size_t bufferSize)
{
    // Verify magic at end of buffer to detect overruns
    uint64_t *magicPtr =
        reinterpret_cast<uint64_t *>((char *)buffer + bufferSize - 2 * sizeof(uint64_t));
    if (magicPtr[0] != node_t::Magic || magicPtr[1] != node_t::Magic) {
        HeapPanic("Buffer overrun detected");
    }
    // Clear magic to avoid double-free issues
    magicPtr[0] = node_t::ClearedMagic;
    magicPtr[1] = node_t::ClearedMagic;
}

static bool
bin_contains(const bin_t *b, const node_t *target)
{
    for (auto *n = b->head; n; n = n->next)
        if (n == target)
            return true;
    return false;
}

// ========================================================
// Init
// ========================================================
void
HeapCpp::Init()
{
    for (size_t i = 0; i < HEAP_BIN_COUNT; i++) {
        Heap.bins[i] = &Heap.binMemory[i];
    }

    Mutex = new (MutexBuffer) std::mutex();

    node_t *init_region = reinterpret_cast<node_t *>(Growable->GetData());

    const uint64_t init_payload_sz = (HEAP_INIT_SIZE) - sizeof(node_t) - sizeof(footer_t);

    // Put the initial region into a clean "free + unlinked" state.
    init_region->init_free_node_unlinked(init_payload_sz);

    CreateFooter(init_region);

    // add_node should be the ONLY thing that transitions in_bin from NotInBin->idx.
    const uint64_t bin = GetBinIndex(init_region->actual_size);
    add_node(Heap.bins[bin], init_region, bin);

    const uintptr_t start = reinterpret_cast<uintptr_t>(init_region);
    Heap.start = start;
    Heap.end = start + HEAP_INIT_SIZE;

    Initialized = true;
}

// ========================================================
// Allocate
// ========================================================
void *
HeapCpp::Allocate(size_t user_size)
{
    if (!Initialized) {
        VectorStaticInit();
        InitGlobalHeap();
    }

    // Round up to nearest 16 bytes
    auto actual_size = (user_size + 0xF) & ~0xF;

    // Tail canary space (two u64)
    actual_size += 16;

    std::unique_lock<std::mutex> lock(*Mutex);
    auto TryOnce = [&]() -> void * {
        uint64_t index = GetBinIndex(actual_size);
        node_t *found = nullptr;

        for (uint64_t i = index; i < HEAP_BIN_COUNT; ++i) {
            bin_t *b = Heap.bins[i];
            found = get_best_fit(b, actual_size);
            if (found)
                break;
        }

        if (!found)
            return nullptr;

        if (!found->hole) {
            HeapPanic("Found node is not free");
        }

        const uint64_t found_bin_idx = found->in_bin;
        HEAP_ASSERT(found_bin_idx != node_t::NotInBin, "Allocate: found is free but NotInBin");
        HEAP_ASSERT(bin_contains(Heap.bins[found_bin_idx], found),
                    "Allocate: in_bin says it's linked, but list traversal can't find it");
        remove_node(Heap.bins[found_bin_idx], found, found_bin_idx);

        // Optional diagnostic (not authoritative):
        HEAP_ASSERT(found_bin_idx == GetBinIndex(found->actual_size),
                    "Allocate: found is in wrong bin for its size");

        // Split if large enough
        if ((found->actual_size - actual_size) > (overhead + MIN_ALLOC_SZ)) {

            node_t *split =
                reinterpret_cast<node_t *>(reinterpret_cast<char *>(found) + overhead + actual_size);

            HEAP_ASSERT(reinterpret_cast<uintptr_t>(split) % alignof(node_t) == 0, "split misaligned");
            HEAP_ASSERT(
                (char *)split >= (char *)Heap.start && (char *)split + sizeof(node_t) < (char *)Heap.end,
                "split out of heap range");
            HEAP_ASSERT((char *)split >= (char *)found + sizeof(node_t) &&
                            (char *)split < (char *)GetFooter(found),
                        "split not inside found block");

            const uint64_t split_payload_sz = found->actual_size - actual_size - overhead;

            // IMPORTANT: initialize ALL free-node metadata (incl in_bin) BEFORE add_node.
            split->init_free_node_unlinked(split_payload_sz);
            CreateFooter(split);

            const auto in_bin = GetBinIndex(split_payload_sz);
            add_node(Heap.bins[in_bin], split, in_bin);

            // shrink found to requested size (+ canary already included)
            found->user_size = static_cast<uint64_t>(user_size);
            found->actual_size = static_cast<uint64_t>(actual_size);
            CreateFooter(found);
        } else {
            CreateFooter(found);
        }

        // Mark allocated (and no longer a freelist node)
        found->hole = 0;
        found->magic = node_t::Magic;

        // You can clear links for tidiness, but they're allocator-owned now anyway.
        found->next = nullptr;
        found->prev = nullptr;

        char *user = (char *)found + sizeof(node_t);
        SetMagicAtEnd(user, found->actual_size);
        return user;
    };

    void *res = TryOnce();
    if (!res) {
        if (Expand(actual_size * 2)) {
            res = TryOnce();
        }
    }

    node_t *wild = GetWilderness();
    if (wild->actual_size < MIN_WILDERNESS) {
        if (!Expand(PAGE_SIZE))
            return nullptr;
    } else if (wild->actual_size > MAX_WILDERNESS) {
        Contract(PAGE_SIZE);
    }

    if (res && (reinterpret_cast<uintptr_t>(res) % 16 != 0)) {
        HeapPanic("Memory allocation is not aligned");
    }

    Stats.Allocations++;
    Stats.BytesAllocated += actual_size;
    return res;
}

// ========================================================
// Deallocate
// ========================================================
static inline bool
ptr_in_heap(const heap_t &H, const void *p)
{
    auto u = reinterpret_cast<uintptr_t>(p);
    return u >= H.start && u < H.end;
}

static inline node_t *
safe_prev_node(const heap_t &H, node_t *head)
{
    if (reinterpret_cast<uintptr_t>(head) == H.start)
        return nullptr;

    // footer immediately before head:
    auto *prev_foot = reinterpret_cast<footer_t *>(reinterpret_cast<char *>(head) - sizeof(footer_t));

    node_t *prev = prev_foot->header;
    if (!prev)
        return nullptr;

    if (!ptr_in_heap(H, prev))
        return nullptr;

    // Optional: sanity that prev's footer points back to prev
    // (catches random pointers early)
    auto *foot = reinterpret_cast<footer_t *>(reinterpret_cast<char *>(prev) + sizeof(node_t) +
                                              prev->actual_size);
    if (!ptr_in_heap(H, foot) || foot->header != prev)
        return nullptr;

    return prev;
}

static inline node_t *
safe_next_node(const heap_t &H, node_t *head, footer_t *head_foot)
{
    auto *next = reinterpret_cast<node_t *>(reinterpret_cast<char *>(head_foot) + sizeof(footer_t));

    // next begins at heap end => no next
    if (!ptr_in_heap(H, next))
        return nullptr;

    // Optional: basic sanity
    if (!ptr_in_heap(H, reinterpret_cast<char *>(next) + sizeof(node_t)))
        return nullptr;

    return next;
}

void
HeapCpp::Deallocate(void *ptr)
{
    if (!ptr)
        return;

    std::unique_lock<std::mutex> lock(*Mutex);
    assert(Initialized);

    node_t *head = (node_t *)((char *)ptr - sizeof(node_t));

    Stats.BytesFreed += head->actual_size;
    Stats.Frees++;

    if (head->magic != node_t::Magic) {
        HeapPanic("Invalid node magic");
    }

    VerifyAndClearMagicAtEnd(ptr, head->actual_size);

    // Identify neighbors in the heap layout (SAFELY)
    footer_t *head_foot = GetFooter(head);
    node_t *prev = safe_prev_node(Heap, head);
    node_t *next = safe_next_node(Heap, head, head_foot);

    // Coalesce with prev if free
    if (prev && prev->hole) {
        const uint64_t idx = prev->in_bin;
        HEAP_ASSERT(idx != node_t::NotInBin, "Deallocate: prev free but NotInBin");
        HEAP_ASSERT(bin_contains(Heap.bins[idx], prev), "Deallocate: prev in_bin mismatch vs list");
        remove_node(Heap.bins[idx], prev, idx);

        // merge
        prev->actual_size += overhead + head->actual_size;
        CreateFooter(prev);
        head = prev;

        // Update head_foot because head changed
        head_foot = GetFooter(head);
    }

    // Coalesce with next if free
    if (next && next->hole) {
        const uint64_t idx = next->in_bin;
        HEAP_ASSERT(idx != node_t::NotInBin, "Deallocate: next free but NotInBin");
        HEAP_ASSERT(bin_contains(Heap.bins[idx], next), "Deallocate: next in_bin mismatch vs list");
        remove_node(Heap.bins[idx], next, idx);

        head->actual_size += overhead + next->actual_size;

        // optional clear
        footer_t *old_foot = GetFooter(next);
        old_foot->header = nullptr;
        next->user_size = 0;
        next->actual_size = 0;
        next->hole = 0;

        CreateFooter(head);
    }

    // Now (coalesced) head becomes a free unlinked node; init metadata BEFORE add_node.
    const uint64_t final_payload_sz = head->actual_size;
    head->init_free_node_unlinked(final_payload_sz);
    CreateFooter(head);

    const auto in_bin = GetBinIndex(head->actual_size);
    add_node(Heap.bins[in_bin], head, in_bin);
}

// ========================================================
// Expand
// ========================================================
bool
HeapCpp::Expand(size_t deltaSizeBytes)
{
    // Mutex must be held

    const auto growSizeBytes = std::max(size_t(GrowByAmtBytes), deltaSizeBytes * 2);
    const auto actual_size = (growSizeBytes + 0xFFF) & ~0xFFF;

    void *new_end = (char *)Heap.end + actual_size;

    Growable->GrowVectorByAmount(actual_size);

    node_t *wild = GetWilderness();

    if (wild->hole) {
        assert_linked_in(wild, wild->in_bin, "Expand: wilderness not linked in expected bin");
        remove_node(Heap.bins[wild->in_bin], wild, wild->in_bin);

        assert_free_unlinked_for_resize(wild, "Expand: wilderness not free/unlinked for resize");
        wild->actual_size += actual_size;
        // wilderness is free + unlinked right now; normalize metadata
        wild->init_free_node_unlinked(wild->actual_size);
        CreateFooter(wild);

        const auto in_bin = GetBinIndex(wild->actual_size);
        add_node(Heap.bins[in_bin], wild, in_bin);

    } else {
        node_t *new_free_block = (node_t *)Heap.end;
        const uint64_t payload_sz = static_cast<uint64_t>(actual_size - overhead);

        new_free_block->init_free_node_unlinked(payload_sz);
        CreateFooter(new_free_block);

        const auto in_bin = GetBinIndex(new_free_block->actual_size);
        add_node(Heap.bins[in_bin], new_free_block, in_bin);
    }

    Heap.end = reinterpret_cast<uintptr_t>(new_end);
    return true;
}

// ========================================================
// Contract (unchanged logic; shown with no in_bin calls)
// ========================================================
bool
HeapCpp::Contract(size_t actual_size)
{
    // Mutex must be held

    actual_size = (actual_size + 0xFFF) & ~0xFFF;

    node_t *wild = GetWilderness();

    // Only contract if wilderness is free; otherwise youre chopping a live block.
    if (!wild->hole) {
        return false;
    }

    if (wild->actual_size < actual_size) {
        return false;
    }

    // Unlink wilderness from its current bin BEFORE changing actual_size.
    const uint64_t old_bin = wild->in_bin;
    HEAP_ASSERT(old_bin != node_t::NotInBin, "Contract: wilderness free but NotInBin");
    HEAP_ASSERT(bin_contains(Heap.bins[old_bin], wild), "Contract: wilderness in_bin mismatch vs list");
    remove_node(Heap.bins[old_bin], wild, old_bin);

    // Now safe to mutate actual_size.
    wild->actual_size -= static_cast<uint64_t>(actual_size);

    // Normalize metadata for a free, unlinked node.
    wild->init_free_node_unlinked(wild->actual_size);
    CreateFooter(wild);

    // Reinsert into correct bin for new size.
    const uint64_t new_bin = GetBinIndex(wild->actual_size);
    add_node(Heap.bins[new_bin], wild, new_bin);

    // Shrink heap end pointer.
    Heap.end -= actual_size;
    return true;
}

size_t
HeapCpp::CountAllocations() const
{
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
        current_address += sizeof(node_t) + current_node->actual_size + sizeof(footer_t);
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
uint64_t
HeapCpp::GetBinIndex(size_t size)
{
    int index = 0;
    constexpr auto minAlignment = 16;
    size = size < minAlignment ? minAlignment : size;

    while (size >>= 1)
        index++;
    index -= 4;

    if (index > HEAP_BIN_MAX_IDX) {
        assert(false);
        index = HEAP_BIN_MAX_IDX;
    }
    return index;
}

// ========================================================
// this function will create a footer given a node
// the node's size must be set to the correct value!
// ========================================================
void
HeapCpp::CreateFooter(node_t *head)
{
    footer_t *foot = GetFooter(head);
    foot->header = head;
}

// ========================================================
// this function will get the footer pointer given a node
// ========================================================
footer_t *
HeapCpp::GetFooter(node_t *head)
{
    return (footer_t *)((char *)head + sizeof(node_t) + head->actual_size);
}

// ========================================================
// this function will get the wilderness node given a
// heap struct pointer
//
// NOTE: this function banks on the heap's end field being
// correct, it simply uses the footer at the end of the
// heap because that is always the wilderness
// ========================================================
node_t *
HeapCpp::GetWilderness()
{
    footer_t *wild_foot = (footer_t *)((char *)Heap.end - sizeof(footer_t));
    return wild_foot->header;
}

//////////////////////////////////////////////////////////////////////////

void *
CppMalloc(size_t size)
{
    auto &globalHeap = GlobalHeap();
    auto *res = globalHeap.Allocate(size);

    // Output res in hex to debug console in Visual Studio
    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "malloc(%zu) = %p\n", size, res);
        OutputDebugStringA(buffer);
    }

    return res;
}

void
CppFree(void *ptr)
{
    auto &globalHeap = GlobalHeap();

    // Output ptr in hex to debug console in Visual Studio
    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "free(%p)\n", ptr);
        OutputDebugStringA(buffer);
    }

    globalHeap.Deallocate(ptr);
}

void *
CppRealloc(void *ptr, size_t newUserSize, bool zeroNew)
{
    if (!ptr) {
        return CppMalloc(newUserSize);
    }

    if (newUserSize == 0) {
        CppFree(ptr);
        return nullptr;
    }

    void *newPtr = CppMalloc(newUserSize);
    if (!newPtr) {
        return nullptr;
    }

    const auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(ptr) - sizeof(node_t));
    const size_t oldUsableSize = node->user_size;
    const size_t copyUserSize = std::min(oldUsableSize, newUserSize);
    std::memcpy(newPtr, ptr, copyUserSize);

    // Zero newly extended region if requested
    if (zeroNew && newUserSize > oldUsableSize) {
        std::memset(static_cast<char *>(newPtr) + oldUsableSize, 0, newUserSize - oldUsableSize);
    }

    CppFree(ptr);
    return newPtr;
}

// Override global malloc, free, and realloc functions
extern "C" {

__declspec(restrict) void *
malloc(size_t size)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppMalloc(size);
    }

    return SysMalloc(size);
}

__declspec(restrict) void *
calloc(size_t num, size_t size)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        void *p = CppMalloc(num * size);
        if (p)
            std::memset(p, 0, num * size);
        return p;
    }
    return SysCalloc(num, size);
}

__declspec(restrict) void *
realloc(void *ptr, size_t newSize)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(ptr, newSize, false);
    }
    return SysRealloc(ptr, newSize);
}

extern "C" void
free(void *ptr)
{
    EarlyInit_SafeMode_NoCRT();

    if (!ptr) {
        return;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        CppFree(ptr);
        return;
    }

    // SysHeap bypass path: always free directly.
    SysFree(ptr);
}


extern "C" __declspec(restrict) void *
aligned_alloc(size_t /*alignment*/, size_t size)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        // Fancy path: keep your existing behavior (or real aligned logic).
        auto p = CppMalloc(size);
        // If you want, keep the assert here for fancy-only.
        // assert(reinterpret_cast<uintptr_t>(p) % alignment == 0);
        return p;
    }

    // SysHeap bypass path: ignore alignment request.
    return SysMalloc(size);
}


char *
strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char *d = (char *)malloc(len);
    if (d)
        memcpy(d, s, len);
    return d;
}

__declspec(restrict) char *
strndup(const char *s, size_t n)
{
    size_t len = strnlen(s, n);
    char *d = (char *)malloc(len + 1);
    if (d) {
        memcpy(d, s, len);
        d[len] = '\0';
    }
    return d;
}

__declspec(restrict) char *
realpath(const char *fname, char *resolved_name)
{
    return _fullpath(resolved_name, fname, _MAX_PATH);
}

#ifdef _DEBUG

void *
_malloc_dbg(size_t size, int blockType, const char *filename, int line)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppMalloc(size);
    }
    return malloc(size);
}

void *
_calloc_dbg(size_t num, size_t size, int blockType, const char *filename, int line)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        void *p = CppMalloc(num * size);
        if (p)
            std::memset(p, 0, num * size);
        return p;
    }
    return calloc(num, size);
}

__declspec(noinline) void *
_realloc_dbg(void *block, size_t requested_size, int block_use, const char *file, int line)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(block, requested_size, false);
    }

    return realloc(block, requested_size);
}

__declspec(noinline) void *
_recalloc_dbg(void *block, size_t count, size_t element_size, int /*block_use*/, const char * /*file*/, int /*line*/)
{
    // Best-effort implementation.
    // Fancy heap path supports explicit zeroing of newly grown bytes.
    // System heap path does not know the prior size, so we fall back to realloc.

    if (element_size != 0 && count > (SIZE_MAX / element_size)) {
        return nullptr;
    }

    size_t newBytes = count * element_size;

    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(block, newBytes, true);
    }

    return SysRealloc(block, newBytes);
}

__declspec(noinline) void *
_expand_dbg(void *, size_t, int, const char *, int)
{
    errno = ENOMEM;
    return nullptr;
}

void
_free_dbg(void *ptr, int blockType)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        CppFree(ptr);
        return;
    }
    free(ptr);
}

__declspec(noinline) size_t
_msize_dbg(void *block, int /*block_use*/)
{
    if (!block) {
        return 0;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(block) - sizeof(node_t));
        return node->user_size;
    }

    SIZE_T s = HeapSize(ProcHeap(), 0, block);
    if (s == (SIZE_T)-1) {
        return 0;
    }
    return static_cast<size_t>(s);
}


#endif

} // extern "C"

void
operator delete(void *ptr)
{
    free(ptr);
}

void
operator delete[](void *ptr)
{
    free(ptr);
}

void *
operator new(size_t size)
{
    void *p = malloc(size);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void *
operator new[](size_t size)
{
    return operator new(size);
}

void *
operator new(size_t n, std::align_val_t al)
{
    return aligned_alloc(static_cast<size_t>(al), n);
}

void *
operator new[](size_t n, std::align_val_t al)
{
    return aligned_alloc(static_cast<size_t>(al), n);
}

void *
operator new(size_t count, const std::nothrow_t &) noexcept
{
    return malloc(count);
}

void *
operator new[](size_t count, const std::nothrow_t &) noexcept
{
    return malloc(count);
}

void *
operator new(size_t count, std::align_val_t al, const std::nothrow_t &) noexcept
{
    return aligned_alloc(static_cast<size_t>(al), count);
}

void *
operator new[](size_t count, std::align_val_t al, const std::nothrow_t &) noexcept
{
    return aligned_alloc(static_cast<size_t>(al), count);
}
