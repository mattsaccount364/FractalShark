#include "include/HeapCpp.h" // Include your wrapper header
#include "../DbgHeap.h"
#include "../EarlyCommandLine.h"
#include "include/HeapPanic.h"
#include "include/llist.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <new>
#include <type_traits>

#include "../Vectors.h"
#include "Environment.h"

#ifdef _MSC_VER
#pragma optimize("", off)
#endif

// Portable compiler attribute macros
#ifdef _MSC_VER
#define FS_RESTRICT __declspec(restrict)
#define FS_NOINLINE __declspec(noinline)
#else
#define __forceinline inline __attribute__((always_inline))
#define FS_RESTRICT __attribute__((malloc))
#define FS_NOINLINE __attribute__((noinline))
#define sprintf_s(buffer, ...) snprintf((buffer), sizeof(buffer), __VA_ARGS__)
#endif

//
// It'd be nice to call RegisterHeapCleanup on first use but atexit in VS2026
// appears to have a table with some stuff in it that's not initialized yet. So
// if we try to register the cleanup here, it crashes.  The result is that we
// see some "leaks" because we intercept all allocations but miss some frees by
// registering later.
//

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

// --------------------------------------------------------
// fs_alloc scaffolding: early-init segment + manual dtor list
// --------------------------------------------------------
//
// Overriding / intercepting allocation can cause the CRT to call into us
// during its own initialization. We therefore place HeapCpp in a special
// init segment and capture its destructor without using atexit early.
//

#ifdef _MSC_VER
#pragma section(".fs_alloc", read, write)
#endif

#ifdef _MSC_VER
using PF = void(__cdecl *)(void);
static int cxpf = 0; // number of destructors we need to call
static PF pfx[200];  // pointers to destructors.

#pragma section(".fs_alloc$a", read)
__declspec(allocate(".fs_alloc$a")) const PF InitSegStart = (PF)1;

#pragma section(".fs_alloc$z", read)
__declspec(allocate(".fs_alloc$z")) const PF InitSegEnd = (PF)1;
#endif

#ifdef _MSC_VER
// Called by MSVC for objects in our init_seg (instead of atexit)
static int __cdecl myexit(PF pf)
{
    if (cxpf >= 200) {
        HeapPanic("Static destructor limit exceeded");
    }
    pfx[cxpf++] = pf;
    return 0;
}

// by default, goes into a read-only section
#pragma warning(disable : 4075)
#pragma init_seg(".fs_alloc$m", myexit)
#endif

FancyHeap EnableFractalSharkHeap;

static constexpr auto PAGE_SIZE = 0x1000;
static constexpr size_t TailCanaryBytes = 16;

// sprintf etc use heap allocations in debug mode, so we can't use them here.
[[maybe_unused]] static constexpr bool LimitedDebugOut = !FractalSharkDebug;
static constexpr bool FullDebugOut = false;

// --------------------------------------------------------
// System heap (Win32 process heap) forwarding
// --------------------------------------------------------

// In 'safemode' or when the fancy heap is disabled, we bypass the CRT allocator
// completely and use the Win32 process heap. This avoids GetProcAddress and
// works safely during very early initialization (TLS/CRT init).

static HeapCpp gHeap;

static __forceinline void *
SysMalloc(size_t n)
{
    if (n == 0) {
        n = 1;
    }
    return Environment::SystemHeapAlloc(n);
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
    return Environment::SystemHeapAllocZeroed(total);
}

static __forceinline void
SysFree(void *p)
{
    if (!p) {
        return;
    }
    Environment::SystemHeapFree(p);
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
    return Environment::SystemHeapRealloc(p, n);
}

[[maybe_unused]] static __forceinline bool
IsPow2(size_t x)
{
    return x && ((x & (x - 1)) == 0);
}

static __forceinline bool
AddSize(size_t a, size_t b, size_t &result)
{
    if (a > SIZE_MAX - b) {
        return false;
    }

    result = a + b;
    return true;
}

static __forceinline bool
AlignUpAddress(uintptr_t value, size_t alignment, uintptr_t &result)
{
    if (alignment == 0 || !IsPow2(alignment)) {
        return false;
    }

    const uintptr_t mask = static_cast<uintptr_t>(alignment - 1);
    if (value > UINTPTR_MAX - mask) {
        return false;
    }

    result = (value + mask) & ~mask;
    return true;
}

static __forceinline bool
RoundUserSize(size_t userSize, size_t &actualSize)
{
    if (userSize > SIZE_MAX - TailCanaryBytes) {
        return false;
    }

    const size_t withCanary = userSize + TailCanaryBytes;
    if (withCanary > SIZE_MAX - (HeapAlignment - 1)) {
        return false;
    }

    actualSize = (withCanary + (HeapAlignment - 1)) & ~(HeapAlignment - 1);
    return true;
}

void
CleanupHeap()
{
    gHeap.~HeapCpp();
}

void
Environment::RegisterHeapCleanup()
{
    if (EnableFractalSharkHeap == FancyHeap::Unknown) {
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
#ifdef _WIN32
        atexit(CleanupHeap);
#endif
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

    // Thread-safe single-initialization guard using atomic CAS.
    // Can't use std::call_once or std::mutex - those may allocate, and this
    // runs before the heap is ready.
    static volatile int32_t init_lock = 0;
    while (Environment::InterlockedCAS32(&init_lock, 1, 0) != 0) {
        // Another thread is initializing - spin until done.
        // After init, Initialized==true and we return early above on next call.
        Environment::SleepMs(0);
    }

    auto &globalHeap = GlobalHeap();

    if (globalHeap.Initialized) {
        Environment::InterlockedCAS32(&init_lock, 0, 1);
        return;
    }

    static_assert(sizeof(GrowableVector<uint8_t>) <= GrowableVectorSize);

    globalHeap.Growable =
        std::construct_at(reinterpret_cast<GrowableVector<uint8_t> *>(globalHeap.GrowableVectorMemory),
                          AddPointOptions::EnableWithoutSave,
                          L"HeapFile.bin");

    globalHeap.Growable->MutableResize(GrowByAmtBytes);
    globalHeap.Init();

    Environment::InterlockedCAS32(&init_lock, 0, 1);
}

HeapCpp::HeapCpp()
{
#ifdef _WIN32
    if (Environment::IsDebuggerAttached()) {
        char buffer[256];
        sprintf_s(buffer, "HeapCpp Constructor called\n");
        Environment::DebugOutput(buffer);
    }
#endif

    // It's very important that we don't do anything here, because this heap
    // may be used before its static constructor is called.
}

HeapCpp::~HeapCpp()
{
    static bool destroyed = false;
    if (destroyed || !Initialized) {
        return;
    }
    destroyed = true;

    // So this is pretty excellent.  We don't want to touch anything here,
    // because this heap may still be used after it's destroyed.

#ifdef _WIN32
    auto totalAllocs = CountAllocations();

    // Check if debugger attached.  Note this isn't necessarily the
    // end of the program, but it's a good place to check because one would
    // expect that it's close.
    if (Environment::IsDebuggerAttached()) {
        // Output totalAllocs to debug console in Visual Studio
        char buffer[256];
        sprintf_s(buffer, "Total allocations remaining = %zu\n", totalAllocs);
        Environment::DebugOutput(buffer);

        // Print Stats
        sprintf_s(buffer, "BytesAllocated = %zu\n", Stats.BytesAllocated);
        Environment::DebugOutput(buffer);

        sprintf_s(buffer, "BytesFreed = %zu\n", Stats.BytesFreed);
        Environment::DebugOutput(buffer);

        sprintf_s(buffer, "Delta bytes allocated = %zu\n", Stats.BytesAllocated - Stats.BytesFreed);
        Environment::DebugOutput(buffer);

        sprintf_s(buffer, "Allocations = %zu\n", Stats.Allocations);
        Environment::DebugOutput(buffer);

        sprintf_s(buffer, "Frees = %zu\n", Stats.Frees);
        Environment::DebugOutput(buffer);

        sprintf_s(buffer, "Delta allocations = %zu\n", Stats.Allocations - Stats.Frees);
        Environment::DebugOutput(buffer);
    }
#endif

    Mutex->~mutex();

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
    HEAP_ASSERT(reinterpret_cast<uintptr_t>(init_region) % alignof(node_t) == 0,
                "Init: heap start misaligned");

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

void *
HeapCpp::FinalizeAllocation(
    node_t *node, size_t userSize, size_t actualSize, bool initializeGeneration, bool checkPoison)
{
    node->user_size = static_cast<uint64_t>(userSize);
    node->actual_size = static_cast<uint64_t>(actualSize);
    CreateFooter(node);

    node->hole = 0;
    node->magic = node_t::Magic;
    node->next = nullptr;
    node->prev = nullptr;

    if (initializeGeneration) {
        node->in_bin = node_t::NotInBin;
        node->in_bin_gen = 0;
    }

    node->head_guard = node_t::HeadGuard;
    node->alloc_gen = node->in_bin_gen;
    node->checksum = node->ComputeChecksum();

    char *user = reinterpret_cast<char *>(node) + sizeof(node_t);

    // Poison spot-check: if this block was previously freed and poisoned,
    // the first 8 bytes of payload should still be 0xCD.
    if (checkPoison && node->poisoned == node_t::WasPoisoned) {
        auto *probe = reinterpret_cast<const uint64_t *>(user);
        if (*probe != 0xCDCDCDCDCDCDCDCDull) {
            char buf[256];
            sprintf_s(buf,
                      "Write-after-free: expected 0xCDCDCDCDCDCDCDCD, got 0x%llX at %p "
                      "(size=%llu)",
                      static_cast<unsigned long long>(*probe),
                      static_cast<void *>(user),
                      static_cast<unsigned long long>(node->actual_size));
            HeapPanic(buf);
        }
    }
    node->poisoned = node_t::NotPoisoned;

    SetMagicAtEnd(user, node->actual_size);
    return user;
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

    size_t actual_size = 0;
    if (!RoundUserSize(user_size, actual_size)) {
        return nullptr;
    }

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

            return FinalizeAllocation(found, user_size, actual_size, false, true);
        } else {
            return FinalizeAllocation(found, user_size, found->actual_size, false, true);
        }
    };

    void *res = TryOnce();
    bool expandAttempted = false;
    bool expandSucceeded = false;
    if (!res) {
        expandAttempted = true;
        expandSucceeded = Expand(actual_size * 2);
        if (expandSucceeded) {
            res = TryOnce();
        }
    }

#ifndef _WIN32
    if (!res) {
        PanicAllocationFailed("Allocate", user_size, actual_size, expandAttempted, expandSucceeded);
    }
#endif
    if (!res) {
        return nullptr;
    }

    node_t *wild = GetWilderness();
    if (wild->actual_size < MIN_WILDERNESS) {
        if (!Expand(PAGE_SIZE))
            return nullptr;
    } else if (wild->actual_size > MAX_WILDERNESS) {
        Contract(PAGE_SIZE);
    }

    if (res && (reinterpret_cast<uintptr_t>(res) % HeapAlignment != 0)) {
        HeapPanic("Memory allocation is not heap-aligned");
    }

    Stats.Allocations++;
    auto *node = reinterpret_cast<node_t *>(reinterpret_cast<char *>(res) - sizeof(node_t));
    Stats.BytesAllocated += node->actual_size;
    return res;
}

void *
HeapCpp::AllocateAligned(size_t user_size, size_t alignment)
{
    if (alignment == 0 || !IsPow2(alignment)) {
        return nullptr;
    }

    if (alignment <= HeapAlignment) {
        return Allocate(user_size);
    }

    if (!Initialized) {
        VectorStaticInit();
        InitGlobalHeap();
    }

    size_t actual_size = 0;
    if (!RoundUserSize(user_size, actual_size)) {
        return nullptr;
    }

    size_t search_size = 0;
    if (!AddSize(actual_size, alignment, search_size) || !AddSize(search_size, overhead, search_size)) {
        return nullptr;
    }

    constexpr size_t MinSplitBytes = overhead + MIN_ALLOC_SZ;

    std::unique_lock<std::mutex> lock(*Mutex);
    auto TryOnce = [&]() -> void * {
        const uint64_t index = GetBinIndex(actual_size);

        for (uint64_t i = index; i < HEAP_BIN_COUNT; ++i) {
            for (node_t *found = Heap.bins[i]->head; found != nullptr; found = found->next) {
                if (found->actual_size < actual_size) {
                    continue;
                }

                HEAP_ASSERT(found->hole, "AllocateAligned: bin node is not free");

                const uintptr_t blockStart = reinterpret_cast<uintptr_t>(found);
                const uintptr_t blockEnd =
                    blockStart + sizeof(node_t) + found->actual_size + sizeof(footer_t);

                uintptr_t alignedUser = 0;
                if (!AlignUpAddress(blockStart + sizeof(node_t), alignment, alignedUser)) {
                    continue;
                }

                uintptr_t alignedHead = alignedUser - sizeof(node_t);
                size_t leadingBytes = static_cast<size_t>(alignedHead - blockStart);

                if (leadingBytes != 0 && leadingBytes < MinSplitBytes) {
                    if (alignedUser > UINTPTR_MAX - alignment) {
                        continue;
                    }
                    alignedUser += alignment;
                    alignedHead = alignedUser - sizeof(node_t);
                    if (alignedHead < blockStart) {
                        continue;
                    }
                    leadingBytes = static_cast<size_t>(alignedHead - blockStart);
                    if (leadingBytes != 0 && leadingBytes < MinSplitBytes) {
                        continue;
                    }
                }

                size_t allocatedBytes = 0;
                if (!AddSize(overhead, actual_size, allocatedBytes)) {
                    continue;
                }
                if (alignedHead < blockStart || alignedHead > blockEnd ||
                    allocatedBytes > static_cast<size_t>(blockEnd - alignedHead)) {
                    continue;
                }

                size_t allocatedActual = actual_size;
                size_t trailingBytes = static_cast<size_t>(blockEnd - alignedHead - allocatedBytes);
                if (trailingBytes != 0 && trailingBytes < MinSplitBytes) {
                    if (!AddSize(allocatedActual, trailingBytes, allocatedActual)) {
                        continue;
                    }
                    trailingBytes = 0;
                }

                const uint64_t foundBinIndex = found->in_bin;
                HEAP_ASSERT(foundBinIndex != node_t::NotInBin,
                            "AllocateAligned: found is free but NotInBin");
                HEAP_ASSERT(bin_contains(Heap.bins[foundBinIndex], found),
                            "AllocateAligned: in_bin says linked, but list traversal can't find it");
                HEAP_ASSERT(foundBinIndex == GetBinIndex(found->actual_size),
                            "AllocateAligned: found is in wrong bin for its size");
                remove_node(Heap.bins[foundBinIndex], found, foundBinIndex);

                node_t *allocated = reinterpret_cast<node_t *>(alignedHead);
                const bool allocatedUsesFoundHeader = allocated == found;

                if (leadingBytes != 0) {
                    const uint64_t leadingPayload = static_cast<uint64_t>(leadingBytes - overhead);
                    found->init_free_node_unlinked(leadingPayload);
                    CreateFooter(found);

                    const auto leadingBin = GetBinIndex(found->actual_size);
                    add_node(Heap.bins[leadingBin], found, leadingBin);
                }

                if (trailingBytes != 0) {
                    const uintptr_t trailingStart = alignedHead + overhead + allocatedActual;
                    node_t *trailing = reinterpret_cast<node_t *>(trailingStart);
                    HEAP_ASSERT(reinterpret_cast<uintptr_t>(trailing) % alignof(node_t) == 0,
                                "AllocateAligned: trailing split misaligned");

                    const uint64_t trailingPayload = static_cast<uint64_t>(trailingBytes - overhead);
                    trailing->init_free_node_unlinked(trailingPayload);
                    CreateFooter(trailing);

                    const auto trailingBin = GetBinIndex(trailing->actual_size);
                    add_node(Heap.bins[trailingBin], trailing, trailingBin);
                }

                return FinalizeAllocation(allocated,
                                          user_size,
                                          allocatedActual,
                                          !allocatedUsesFoundHeader,
                                          allocatedUsesFoundHeader);
            }
        }

        return nullptr;
    };

    void *res = TryOnce();
    bool expandAttempted = false;
    bool expandSucceeded = false;
    if (!res) {
        expandAttempted = true;
        expandSucceeded = Expand(search_size);
        if (expandSucceeded) {
            res = TryOnce();
        }
    }

#ifndef _WIN32
    if (!res) {
        PanicAllocationFailed(
            "AllocateAligned", user_size, actual_size, expandAttempted, expandSucceeded);
    }
#endif
    if (!res) {
        return nullptr;
    }

    node_t *wild = GetWilderness();
    if (wild->actual_size < MIN_WILDERNESS) {
        if (!Expand(PAGE_SIZE))
            return nullptr;
    } else if (wild->actual_size > MAX_WILDERNESS) {
        Contract(PAGE_SIZE);
    }

    if (res && (reinterpret_cast<uintptr_t>(res) % alignment != 0)) {
        HeapPanic("Aligned allocation is not aligned");
    }

    auto *node = reinterpret_cast<node_t *>(reinterpret_cast<char *>(res) - sizeof(node_t));
    Stats.Allocations++;
    Stats.BytesAllocated += node->actual_size;
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
safe_next_node(const heap_t &H, node_t * /*head*/, footer_t *head_foot)
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

    if (!ptr_in_heap(Heap, head)) {
        HeapPanic("Free of pointer outside heap bounds");
    }

    if (head->hole) {
        HeapPanic("Double free detected");
    }

    if (head->magic != node_t::Magic) {
        HeapPanic("Invalid node magic");
    }

    // Verify metadata checksum (catches wild-pointer scribbles on header)
    if (head->checksum != head->ComputeChecksum()) {
        HeapPanic("Node metadata checksum mismatch (header corruption)");
    }

    // Verify head guard (catches backward scribbles from user data)
    if (head->head_guard != node_t::HeadGuard) {
        HeapPanic("Head guard corrupted (buffer underflow)");
    }

    // Verify generation counter (catches use-after-free)
    if (head->alloc_gen != head->in_bin_gen) {
        HeapPanic("Use-after-free detected (generation mismatch)");
    }

    Stats.BytesFreed += head->actual_size;
    Stats.Frees++;

    VerifyAndClearMagicAtEnd(ptr, head->actual_size);

    // Poison user data to catch use-after-free
    std::memset(ptr, 0xCD, head->actual_size);

    // Identify neighbors in the heap layout (SAFELY)
    footer_t *head_foot = GetFooter(head);
    node_t *prev = safe_prev_node(Heap, head);
    node_t *next = safe_next_node(Heap, head, head_foot);

    bool coalesced = false;

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
        coalesced = true;
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
        coalesced = true;
    }

    // Now (coalesced) head becomes a free unlinked node; init metadata BEFORE add_node.
    const uint64_t final_payload_sz = head->actual_size;
    head->init_free_node_unlinked(final_payload_sz);
    CreateFooter(head);

    // Mark poisoned only if block boundaries didn't change (no coalescing).
    // Coalesced blocks include regions that weren't poisoned.
    if (!coalesced) {
        head->poisoned = node_t::WasPoisoned;
    }

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

    // Guard against infinite recursion: if GrowableVector's error path
    // allocates heap memory, it would re-enter Expand via malloc -> HeapCpp.
    static thread_local bool expanding = false;
    if (expanding) {
        HeapPanic("Expand: recursion detected - GrowableVector error path allocated heap");
    }
    expanding = true;

    const auto growSizeBytes = std::max(size_t(GrowByAmtBytes), deltaSizeBytes * 2);
    const auto actual_size = (growSizeBytes + 0xFFF) & ~0xFFF;

    void *new_end = (char *)Heap.end + actual_size;

    Growable->GrowVectorByAmount(actual_size);

    expanding = false;

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
        HEAP_ASSERT(reinterpret_cast<uintptr_t>(new_free_block) % alignof(node_t) == 0,
                    "Expand: new block misaligned");
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

    uintptr_t current_address = Heap.start;

    while (current_address < Heap.end) {
        node_t *current_node = reinterpret_cast<node_t *>(current_address);

        // Validate node before trusting actual_size for traversal
        if (current_node->hole == 0 && current_node->magic != node_t::Magic) {
            HeapPanic("CountAllocations: corrupt allocated node");
        }
        if (current_node->hole == 1 && current_node->magic != node_t::ClearedMagic) {
            HeapPanic("CountAllocations: corrupt free node");
        }
        if (current_node->actual_size == 0 ||
            current_address + sizeof(node_t) + current_node->actual_size + sizeof(footer_t) > Heap.end) {
            HeapPanic("CountAllocations: actual_size out of range");
        }

        if (current_node->hole == 0) {
            count++;
        }

        current_address += sizeof(node_t) + current_node->actual_size + sizeof(footer_t);
    }

    return count;
}

bool
HeapCpp::OwnsPointer(const void *ptr) const
{
    if (!Initialized || ptr == nullptr) {
        return false;
    }

    const auto address = reinterpret_cast<uintptr_t>(ptr);
    return address >= Heap.start && address < Heap.end;
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
    constexpr auto minAlignment = HeapAlignment;
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

[[noreturn]] void
HeapCpp::PanicAllocationFailed(const char *operation,
                               size_t userSize,
                               size_t actualSize,
                               bool expandAttempted,
                               bool expandSucceeded)
{
    const size_t heapBytes = Initialized ? static_cast<size_t>(Heap.end - Heap.start) : 0;
    size_t wildernessBytes = 0;
    if (Initialized && Heap.end > Heap.start) {
        node_t *wild = GetWilderness();
        if (wild != nullptr) {
            wildernessBytes = static_cast<size_t>(wild->actual_size);
        }
    }

    char buf[512];
    sprintf_s(buf,
              "Heap allocation failed: op=%s user=%zu actual=%zu heap=%zu wilderness=%zu "
              "expandAttempted=%d expandSucceeded=%d",
              operation,
              userSize,
              actualSize,
              heapBytes,
              wildernessBytes,
              expandAttempted ? 1 : 0,
              expandSucceeded ? 1 : 0);
    HeapPanic(buf);
}

//////////////////////////////////////////////////////////////////////////

void *
CppMalloc(size_t size)
{
    auto &globalHeap = GlobalHeap();
    auto *res = globalHeap.Allocate(size);
#ifndef _WIN32
    if (res == nullptr && EnableFractalSharkHeap == FancyHeap::Enable) {
        char buf[256];
        sprintf_s(buf, "CppMalloc returned null: size=%zu", size);
        HeapPanic(buf);
    }
#endif

    // Output res in hex to debug console in Visual Studio
    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "malloc(%zu) = %p\n", size, res);
        Environment::DebugOutput(buffer);
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
        Environment::DebugOutput(buffer);
    }

    globalHeap.Deallocate(ptr);
}

static node_t *
GetLiveNodeFromUserPointer(void *ptr, const char *operation)
{
    auto &globalHeap = GlobalHeap();
    auto *node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(ptr) - sizeof(node_t));
    if (!globalHeap.OwnsPointer(node)) {
        HeapPanic(operation);
    }
    if (node->magic != node_t::Magic) {
        HeapPanic(operation);
    }
    if (node->hole) {
        HeapPanic(operation);
    }
    return node;
}

[[maybe_unused]] static size_t
CppUsableSize(void *ptr, const char *operation)
{
    if (ptr == nullptr) {
        return 0;
    }

    return GetLiveNodeFromUserPointer(ptr, operation)->user_size;
}

void *
CppAlignedMalloc(size_t size, size_t alignment)
{
    auto &globalHeap = GlobalHeap();
    auto *res = globalHeap.AllocateAligned(size, alignment);

    if constexpr (FullDebugOut) {
        char buffer[256];
        sprintf(buffer, "aligned_alloc(%zu, %zu) = %p\n", alignment, size, res);
        Environment::DebugOutput(buffer);
    }

    return res;
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
#ifndef _WIN32
        if (EnableFractalSharkHeap == FancyHeap::Enable) {
            char buf[256];
            sprintf_s(buf, "CppRealloc failed: ptr=%p newSize=%zu", ptr, newUserSize);
            HeapPanic(buf);
        }
#endif
        return nullptr;
    }

    const auto node = GetLiveNodeFromUserPointer(ptr, "Realloc: invalid or freed block");
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

void *
CppAlignedRealloc(void *ptr, size_t newUserSize, size_t alignment, bool zeroNew)
{
    if (!ptr) {
        return CppAlignedMalloc(newUserSize, alignment);
    }

    if (newUserSize == 0) {
        CppFree(ptr);
        return nullptr;
    }

    const auto node = GetLiveNodeFromUserPointer(ptr, "Aligned realloc: invalid or freed block");
    const size_t oldUsableSize = node->user_size;

    void *newPtr = CppAlignedMalloc(newUserSize, alignment);
    if (!newPtr) {
#ifndef _WIN32
        if (EnableFractalSharkHeap == FancyHeap::Enable) {
            char buf[256];
            sprintf_s(buf,
                      "CppAlignedRealloc failed: ptr=%p newSize=%zu alignment=%zu",
                      ptr,
                      newUserSize,
                      alignment);
            HeapPanic(buf);
        }
#endif
        return nullptr;
    }

    const size_t copyUserSize = std::min(oldUsableSize, newUserSize);
    std::memcpy(newPtr, ptr, copyUserSize);

    if (zeroNew && newUserSize > oldUsableSize) {
        std::memset(static_cast<char *>(newPtr) + oldUsableSize, 0, newUserSize - oldUsableSize);
    }

    CppFree(ptr);
    return newPtr;
}

// Override global malloc, free, and realloc functions
extern "C" {

FS_RESTRICT void *
malloc(size_t size)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppMalloc(size);
    }

    return SysMalloc(size);
}

FS_RESTRICT void *
calloc(size_t num, size_t size)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        if (size != 0 && num > (SIZE_MAX / size)) {
            return nullptr;
        }
        size_t total = num * size;
        void *p = CppMalloc(total);
        if (p)
            std::memset(p, 0, total);
        return p;
    }
    return SysCalloc(num, size);
}

FS_RESTRICT void *
realloc(void *ptr, size_t newSize)
{
    // Ensure the safemode decision has been made as early as possible (must not allocate).
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(ptr, newSize, false);
    }
    return SysRealloc(ptr, newSize);
}

#ifndef _WIN32
FS_RESTRICT void *
reallocarray(void *ptr, size_t num, size_t size)
{
    if (size != 0 && num > (SIZE_MAX / size)) {
        errno = ENOMEM;
        return nullptr;
    }

    return realloc(ptr, num * size);
}
#endif

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

extern "C" FS_RESTRICT void *
aligned_alloc(size_t alignment, size_t size)
{
    EarlyInit_SafeMode_NoCRT();

    if (alignment == 0 || !IsPow2(alignment)) {
        errno = EINVAL;
        return nullptr;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        auto p = CppAlignedMalloc(size, alignment);
#ifndef _WIN32
        if (p == nullptr) {
            char buf[256];
            sprintf_s(buf, "aligned_alloc returned null: size=%zu alignment=%zu", size, alignment);
            HeapPanic(buf);
        }
#endif
        if (p && (reinterpret_cast<uintptr_t>(p) % alignment != 0)) {
            char buf[256];
            sprintf_s(buf,
                      "aligned_alloc: allocation not aligned to requested boundary "
                      "size=%zu alignment=%zu ptr=%p",
                      size,
                      alignment,
                      p);
            HeapPanic(buf);
        }
        return p;
    }

    // SysHeap bypass path: ignore alignment request.
    return SysMalloc(size);
}

#ifndef _WIN32
int
posix_memalign(void **memptr, size_t alignment, size_t size)
{
    EarlyInit_SafeMode_NoCRT();

    if (alignment < sizeof(void *) || !IsPow2(alignment)) {
        return EINVAL;
    }

    if (EnableFractalSharkHeap != FancyHeap::Enable) {
        return ENOMEM;
    }

    void *ptr = aligned_alloc(alignment, size);
    if (ptr == nullptr) {
        return errno == EINVAL ? EINVAL : ENOMEM;
    }

    *memptr = ptr;
    return 0;
}
#endif

char *
strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char *d = (char *)malloc(len);
    if (d)
        memcpy(d, s, len);
    return d;
}

FS_RESTRICT char *
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

#ifdef _WIN32
FS_RESTRICT char *
realpath(const char *fname, char *resolved_name)
{
    return _fullpath(resolved_name, fname, _MAX_PATH);
}

FS_RESTRICT void *__cdecl _aligned_malloc(size_t size, size_t alignment)
{
    EarlyInit_SafeMode_NoCRT();

    if (alignment == 0 || !IsPow2(alignment)) {
        errno = EINVAL;
        return nullptr;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppAlignedMalloc(size, alignment);
    }

    return SysMalloc(size);
}

void __cdecl _aligned_free(void *ptr) { free(ptr); }

void *__cdecl _aligned_realloc(void *ptr, size_t size, size_t alignment)
{
    EarlyInit_SafeMode_NoCRT();

    if (alignment == 0 || !IsPow2(alignment)) {
        errno = EINVAL;
        return nullptr;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppAlignedRealloc(ptr, size, alignment, false);
    }

    return SysRealloc(ptr, size);
}

size_t __cdecl _aligned_msize(void *ptr, size_t /*alignment*/, size_t /*offset*/)
{
    if (!ptr) {
        return 0;
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppUsableSize(ptr, "_aligned_msize: invalid or freed block");
    }

    return Environment::SystemHeapSize(ptr);
}
#endif

#if defined(_MSC_VER) && defined(_DEBUG)

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
        if (size != 0 && num > (SIZE_MAX / size)) {
            return nullptr;
        }
        size_t total = num * size;
        void *p = CppMalloc(total);
        if (p)
            std::memset(p, 0, total);
        return p;
    }
    return calloc(num, size);
}

FS_NOINLINE void *
_realloc_dbg(void *block, size_t requested_size, int block_use, const char *file, int line)
{
    EarlyInit_SafeMode_NoCRT();

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(block, requested_size, false);
    }

    return realloc(block, requested_size);
}

FS_NOINLINE void *
_recalloc_dbg(void *block,
              size_t count,
              size_t element_size,
              int /*block_use*/,
              const char * /*file*/,
              int /*line*/)
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

FS_NOINLINE void *
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
        return CppUsableSize(block, "_msize_dbg: invalid or freed block");
    }

    return Environment::SystemHeapSize(block);
}

#endif

} // extern "C"

void
operator delete(void *ptr) noexcept
{
    free(ptr);
}

void
operator delete[](void *ptr) noexcept
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
    void *p = aligned_alloc(static_cast<size_t>(al), n);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void *
operator new[](size_t n, std::align_val_t al)
{
    void *p = aligned_alloc(static_cast<size_t>(al), n);
    if (!p)
        throw std::bad_alloc();
    return p;
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

void
operator delete(void *ptr, std::size_t) noexcept
{
    free(ptr);
}

void
operator delete[](void *ptr, std::size_t) noexcept
{
    free(ptr);
}

void
operator delete(void *ptr, std::align_val_t) noexcept
{
    free(ptr);
}

void
operator delete[](void *ptr, std::align_val_t) noexcept
{
    free(ptr);
}

void
operator delete(void *ptr, std::size_t, std::align_val_t) noexcept
{
    free(ptr);
}

void
operator delete[](void *ptr, std::size_t, std::align_val_t) noexcept
{
    free(ptr);
}
