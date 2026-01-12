#include "include\HeapCpp.h" // Include your wrapper header
#include "..\DbgHeap.h"
#include "include\HeapCommandLine.h"
#include "include\HeapPanic.h"
#include "include\llist.h"

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "..\Vectors.h"

#define NOMINMAX
#include <windows.h>

#pragma optimize("", off)


FancyHeap EnableFractalSharkHeap;

static constexpr auto PAGE_SIZE = 0x1000;

// CRT forwarding declarations
extern "C" {

void *__cdecl crt_malloc(size_t);
void *__cdecl crt_calloc(size_t, size_t);
void *__cdecl crt_realloc(void *, size_t);
void __cdecl crt_free(void *);

#ifdef _DEBUG
void *__cdecl crt_malloc_dbg(size_t, int, const char *, int);
void *__cdecl crt_calloc_dbg(size_t, size_t, int, const char *, int);
void *__cdecl crt_realloc_dbg(void *, size_t, int, const char *, int);
void *__cdecl crt_recalloc_dbg(void *, size_t, size_t, int, const char *, int);
void __cdecl crt_free_dbg(void *, int);
size_t __cdecl crt_msize_dbg(void *, int);
#endif
}

// sprintf etc use heap allocations in debug mode, so we can't use them here.
static constexpr bool LimitedDebugOut = !FractalSharkDebug;
static constexpr bool FullDebugOut = false;

// --- CRT forwarding: resolve *all* CRT entrypoints we might need ---
// This supports the overrides you listed: malloc/calloc/realloc/free,
// plus (debug) _malloc_dbg/_calloc_dbg/_realloc_dbg/_recalloc_dbg/_free_dbg/_msize_dbg.
//
// NOTE: operator new/delete are C++ and should NOT be resolved here; they just call
// malloc/free (or aligned_alloc) which dispatches via EnableFractalSharkHeap.
//
// Also note: MSVC's aligned routines are usually _aligned_malloc/_aligned_free rather
// than C11 aligned_alloc. If you want to forward aligned allocations to CRT when
// heap is disabled, resolve those too.

#include <cassert>
#include <cstddef>
#include <windows.h>

// --------------------- release CRT allocator function types ---------------------
using malloc_fn = void *(__cdecl *)(size_t);
using free_fn = void(__cdecl *)(void *);
using realloc_fn = void *(__cdecl *)(void *, size_t);
using calloc_fn = void *(__cdecl *)(size_t, size_t);

// MSVC alignment helpers (recommended if you want true CRT-aligned behavior)
using aligned_malloc_fn = void *(__cdecl *)(size_t size, size_t alignment);
using aligned_free_fn = void(__cdecl *)(void *);

// --------------------- debug CRT allocator function types ----------------------
#ifdef _DEBUG
using malloc_dbg_fn = void *(__cdecl *)(size_t, int, const char *, int);
using calloc_dbg_fn = void *(__cdecl *)(size_t, size_t, int, const char *, int);
using realloc_dbg_fn = void *(__cdecl *)(void *, size_t, int, const char *, int);
using recalloc_dbg_fn = void *(__cdecl *)(void *, size_t, size_t, int, const char *, int);
using free_dbg_fn = void(__cdecl *)(void *, int);
using msize_dbg_fn = size_t(__cdecl *)(void *, int);
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

//
// We create a special section so that constructor does not run automatically.
// Instead we run it manually.
//

#pragma section(".fs_alloc", write)

typedef void(__cdecl *PF)(void);
int cxpf = 0; // number of destructors we need to call
PF pfx[200];  // pointers to destructors.

#pragma section(".fs_alloc$a", read)
__declspec(allocate(".fs_alloc$a")) const PF InitSegStart = (PF)1;

#pragma section(".fs_alloc$z", read)
__declspec(allocate(".fs_alloc$z")) const PF InitSegEnd = (PF)1;

int __cdecl
myexit(PF pf)
{
    pfx[cxpf++] = pf;
    return 0;
}

// by default, goes into a read only section
#pragma warning(disable : 4075)
#pragma init_seg(".fs_alloc$m", myexit)

// #pragma init_seg("fs_alloc$m")
//__declspec(allocate("fs_alloc$m"))
static HeapCpp gHeap;

// --------------------- resolved function pointers ------------------------------
__declspec(allocate(".fs_alloc")) static malloc_fn g_crt_malloc;
__declspec(allocate(".fs_alloc")) static free_fn g_crt_free;
__declspec(allocate(".fs_alloc")) static realloc_fn g_crt_realloc;
__declspec(allocate(".fs_alloc")) static calloc_fn g_crt_calloc;

// Optional but useful if you want CRT-backed aligned allocations when disabled.
__declspec(allocate(".fs_alloc")) static aligned_malloc_fn g_crt_aligned_malloc;
__declspec(allocate(".fs_alloc")) static aligned_free_fn g_crt_aligned_free;

#ifdef _DEBUG
__declspec(allocate(".fs_alloc")) static malloc_dbg_fn g_crt_malloc_dbg;
__declspec(allocate(".fs_alloc")) static calloc_dbg_fn g_crt_calloc_dbg;
__declspec(allocate(".fs_alloc")) static realloc_dbg_fn g_crt_realloc_dbg;
__declspec(allocate(".fs_alloc")) static recalloc_dbg_fn g_crt_recalloc_dbg;
__declspec(allocate(".fs_alloc")) static free_dbg_fn g_crt_free_dbg;
__declspec(allocate(".fs_alloc")) static msize_dbg_fn g_crt_msize_dbg;
#endif

// Check if any of these globals is nullptr
bool
CheckGlobalsInitted()
{
    if (!g_crt_malloc || !g_crt_free || !g_crt_realloc || !g_crt_calloc) {
        return false;
    }

    if (!g_crt_aligned_malloc || !g_crt_aligned_free) {
        return false;
    }

#ifdef _DEBUG
    if (!g_crt_malloc_dbg || !g_crt_calloc_dbg || !g_crt_realloc_dbg ||
        !g_crt_recalloc_dbg || !g_crt_free_dbg || !g_crt_msize_dbg) {
        return false;
    }
#endif

    if (!CheckGlobalsInitted()) {
        return false;
    }

    return true;
}


static FARPROC
MustGetProc(HMODULE m, const char *name)
{
    FARPROC p = GetProcAddress(m, name);
    assert(p && "Missing CRT export");
    return p;
}

// Resolve CRT symbols. Important: use GetModuleHandle/GetProcAddress,
// and do not call malloc/new/etc while resolving.
static void
ResolveCrtAllocators()
{
    // UCRT on modern MSVC; msvcrt.dll on older/other.
    HMODULE crt = GetModuleHandleW(L"ucrtbase.dll");
    if (!crt)
        crt = GetModuleHandleW(L"msvcrt.dll");

    if (!crt) {
        assert(false && "Could not locate CRT module for allocator forwarding");
        return;
    }

    // Release C allocator API
    g_crt_malloc = reinterpret_cast<malloc_fn>(MustGetProc(crt, "malloc"));
    g_crt_free = reinterpret_cast<free_fn>(MustGetProc(crt, "free"));
    g_crt_realloc = reinterpret_cast<realloc_fn>(MustGetProc(crt, "realloc"));
    g_crt_calloc = reinterpret_cast<calloc_fn>(MustGetProc(crt, "calloc"));

    // MSVC aligned allocation APIs (present in ucrtbase/msvcrt in typical configs)
    // If your build doesn't export these, you can drop them or soft-fail them.
    g_crt_aligned_malloc = reinterpret_cast<aligned_malloc_fn>(GetProcAddress(crt, "_aligned_malloc"));
    g_crt_aligned_free = reinterpret_cast<aligned_free_fn>(GetProcAddress(crt, "_aligned_free"));

#ifdef _DEBUG
    HMODULE crt_d = GetModuleHandleW(L"ucrtbased.dll");
    if (!crt_d)
        crt_d = GetModuleHandleW(L"msvcrtd.dll");

    if (!crt_d) {
        assert(false && "Could not locate Debug CRT module for allocator forwarding");
        return;
    }

    // Debug CRT entrypoints. These are not always exported by the same DLL depending
    // on configuration, but in many MSVC debug configurations they resolve from ucrtbase/msvcrt.
    // If they don't exist in your setup, make these optional (null) and avoid calling them.
    g_crt_malloc_dbg = reinterpret_cast<malloc_dbg_fn>(GetProcAddress(crt_d, "_malloc_dbg"));
    g_crt_calloc_dbg = reinterpret_cast<calloc_dbg_fn>(GetProcAddress(crt_d, "_calloc_dbg"));
    g_crt_realloc_dbg = reinterpret_cast<realloc_dbg_fn>(GetProcAddress(crt_d, "_realloc_dbg"));
    g_crt_recalloc_dbg = reinterpret_cast<recalloc_dbg_fn>(GetProcAddress(crt_d, "_recalloc_dbg"));
    g_crt_free_dbg = reinterpret_cast<free_dbg_fn>(GetProcAddress(crt_d, "_free_dbg"));
    g_crt_msize_dbg = reinterpret_cast<msize_dbg_fn>(GetProcAddress(crt_d, "_msize_dbg"));

    // If you *require* the dbg exports, hard-assert them:
    // assert(g_crt_malloc_dbg && g_crt_calloc_dbg && g_crt_realloc_dbg &&
    //        g_crt_recalloc_dbg && g_crt_free_dbg && g_crt_msize_dbg);
#endif

    assert(g_crt_malloc && g_crt_free && g_crt_realloc && g_crt_calloc);
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
    const uint64_t bin = GetBinIndex(init_region->size);
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
HeapCpp::Allocate(size_t size)
{
    if (!Initialized) {
        VectorStaticInit();
        InitGlobalHeap();
    }

    // Round up to nearest 16 bytes
    size = (size + 0xF) & ~0xF;

    // Tail canary space (two u64)
    size += 16;

    std::unique_lock<std::mutex> lock(*Mutex);
    auto TryOnce = [&]() -> void * {
        uint64_t index = GetBinIndex(size);
        node_t *found = nullptr;

        for (uint64_t i = index; i < HEAP_BIN_COUNT; ++i) {
            bin_t *b = Heap.bins[i];
            found = get_best_fit(b, size);
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
        HEAP_ASSERT(found_bin_idx == GetBinIndex(found->size),
                    "Allocate: found is in wrong bin for its size");

        // Split if large enough
        if ((found->size - size) > (overhead + MIN_ALLOC_SZ)) {

            node_t *split =
                reinterpret_cast<node_t *>(reinterpret_cast<char *>(found) + overhead + size);

            HEAP_ASSERT(reinterpret_cast<uintptr_t>(split) % alignof(node_t) == 0, "split misaligned");
            HEAP_ASSERT(
                (char *)split >= (char *)Heap.start && (char *)split + sizeof(node_t) < (char *)Heap.end,
                "split out of heap range");
            HEAP_ASSERT((char *)split >= (char *)found + sizeof(node_t) &&
                            (char *)split < (char *)GetFooter(found),
                        "split not inside found block");

            const uint64_t split_payload_sz = found->size - size - overhead;

            // IMPORTANT: initialize ALL free-node metadata (incl in_bin) BEFORE add_node.
            split->init_free_node_unlinked(split_payload_sz);
            CreateFooter(split);

            const auto in_bin = GetBinIndex(split_payload_sz);
            add_node(Heap.bins[in_bin], split, in_bin);

            // shrink found to requested size (+ canary already included)
            found->size = static_cast<uint64_t>(size);
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
        SetMagicAtEnd(user, found->size);
        return user;
    };

    void *res = TryOnce();
    if (!res) {
        if (Expand(size * 2)) {
            res = TryOnce();
        }
    }

    node_t *wild = GetWilderness();
    if (wild->size < MIN_WILDERNESS) {
        if (!Expand(PAGE_SIZE))
            return nullptr;
    } else if (wild->size > MAX_WILDERNESS) {
        Contract(PAGE_SIZE);
    }

    if (res && (reinterpret_cast<uintptr_t>(res) % 16 != 0)) {
        HeapPanic("Memory allocation is not aligned");
    }

    Stats.Allocations++;
    Stats.BytesAllocated += size;
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
    auto *foot =
        reinterpret_cast<footer_t *>(reinterpret_cast<char *>(prev) + sizeof(node_t) + prev->size);
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

    Stats.BytesFreed += head->size;
    Stats.Frees++;

    if (head->magic != node_t::Magic) {
        HeapPanic("Invalid node magic");
    }

    VerifyAndClearMagicAtEnd(ptr, head->size);

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
        prev->size += overhead + head->size;
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

        head->size += overhead + next->size;

        // optional clear
        footer_t *old_foot = GetFooter(next);
        old_foot->header = nullptr;
        next->size = 0;
        next->hole = 0;

        CreateFooter(head);
    }

    // Now (coalesced) head becomes a free unlinked node; init metadata BEFORE add_node.
    const uint64_t final_payload_sz = head->size;
    head->init_free_node_unlinked(final_payload_sz);
    CreateFooter(head);

    const auto in_bin = GetBinIndex(head->size);
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
    const auto size = (growSizeBytes + 0xFFF) & ~0xFFF;

    void *new_end = (char *)Heap.end + size;

    Growable->GrowVectorByAmount(size);

    node_t *wild = GetWilderness();

    if (wild->hole) {
        assert_linked_in(wild, wild->in_bin, "Expand: wilderness not linked in expected bin");
        remove_node(Heap.bins[wild->in_bin], wild, wild->in_bin);

        assert_free_unlinked_for_resize(wild, "Expand: wilderness not free/unlinked for resize");
        wild->size += size;
        // wilderness is free + unlinked right now; normalize metadata
        wild->init_free_node_unlinked(wild->size);
        CreateFooter(wild);

        const auto in_bin = GetBinIndex(wild->size);
        add_node(Heap.bins[in_bin], wild, in_bin);

    } else {
        node_t *new_free_block = (node_t *)Heap.end;
        const uint64_t payload_sz = static_cast<uint64_t>(size - overhead);

        new_free_block->init_free_node_unlinked(payload_sz);
        CreateFooter(new_free_block);

        const auto in_bin = GetBinIndex(new_free_block->size);
        add_node(Heap.bins[in_bin], new_free_block, in_bin);
    }

    Heap.end = reinterpret_cast<uintptr_t>(new_end);
    return true;
}

// ========================================================
// Contract (unchanged logic; shown with no in_bin calls)
// ========================================================
bool
HeapCpp::Contract(size_t size)
{
    // Mutex must be held

    size = (size + 0xFFF) & ~0xFFF;

    node_t *wild = GetWilderness();

    // Only contract if wilderness is free; otherwise you’re chopping a live block.
    if (!wild->hole) {
        return false;
    }

    if (wild->size < size) {
        return false;
    }

    // Unlink wilderness from its current bin BEFORE changing size.
    const uint64_t old_bin = wild->in_bin;
    HEAP_ASSERT(old_bin != node_t::NotInBin, "Contract: wilderness free but NotInBin");
    HEAP_ASSERT(bin_contains(Heap.bins[old_bin], wild), "Contract: wilderness in_bin mismatch vs list");
    remove_node(Heap.bins[old_bin], wild, old_bin);

    // Now safe to mutate size.
    wild->size -= static_cast<uint64_t>(size);

    // Normalize metadata for a free, unlinked node.
    wild->init_free_node_unlinked(wild->size);
    CreateFooter(wild);

    // Reinsert into correct bin for new size.
    const uint64_t new_bin = GetBinIndex(wild->size);
    add_node(Heap.bins[new_bin], wild, new_bin);

    // Shrink heap end pointer.
    Heap.end -= size;
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
CppRealloc(void *ptr, size_t newSize, bool zeroNew)
{
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
    const auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(ptr) - sizeof(node_t));

    // Assume the size is stored somehow, or manage separately in the Heap class
    std::memcpy(newPtr, ptr, std::min(node->size, newSize));
    CppFree(ptr);

    // Erase any new memory if requested
    if (zeroNew && newSize > node->size) {
        std::memset(static_cast<char *>(newPtr) + node->size, 0, newSize - node->size);
    }

    return newPtr;
}

// Override global malloc, free, and realloc functions
extern "C" {

__declspec(restrict) void *
malloc(size_t size)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppMalloc(size);
    }

    return g_crt_malloc(size);
}

__declspec(restrict) void *
calloc(size_t num, size_t size)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        void *p = CppMalloc(num * size);
        if (p)
            std::memset(p, 0, num * size);
        return p;
    }
    return g_crt_calloc(num, size);
}

__declspec(restrict) void *
realloc(void *ptr, size_t newSize)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(ptr, newSize, false);
    }
    return g_crt_realloc(ptr, newSize);
}

void
free(void *ptr)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        CppFree(ptr);
        return;
    }
    g_crt_free(ptr);
}

__declspec(restrict) void *
aligned_alloc(size_t alignment, size_t size)
{
    void *p = malloc(size);
    assert(reinterpret_cast<uintptr_t>(p) % alignment == 0);
    return p;
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
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppMalloc(size);
    }
    return g_crt_malloc_dbg(size, blockType, filename, line);
}

void *
_calloc_dbg(size_t num, size_t size, int blockType, const char *filename, int line)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        void *p = CppMalloc(num * size);
        if (p)
            std::memset(p, 0, num * size);
        return p;
    }
    return g_crt_calloc_dbg(num, size, blockType, filename, line);
}

__declspec(noinline) void *
_realloc_dbg(void *block, size_t requested_size, int block_use, const char *file, int line)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(block, requested_size, false);
    }

    return g_crt_realloc_dbg(block, requested_size, block_use, file, line);
}

__declspec(noinline) void *
_recalloc_dbg(void *block, size_t count, size_t element_size, int block_use, const char *file, int line)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        return CppRealloc(block, count * element_size, true);
    }

    return g_crt_recalloc_dbg(block, count, element_size, block_use, file, line);
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
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        CppFree(ptr);
        return;
    }
    g_crt_free_dbg(ptr, blockType);
}

__declspec(noinline) size_t
_msize_dbg(void *block, int block_use)
{
    if (!CheckGlobalsInitted()) {
        ResolveCrtAllocators();
        EarlyInit_SafeMode_NoCRT();
    }

    if (!block)
        return 0;

    if (EnableFractalSharkHeap == FancyHeap::Enable) {
        auto node = reinterpret_cast<node_t *>(reinterpret_cast<uintptr_t>(block) - sizeof(node_t));
        return node->size;
    }

    return g_crt_msize_dbg(block, block_use);
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
