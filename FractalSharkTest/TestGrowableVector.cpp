#include "TestFramework.h"

#include "Environment.h"
#include "Vectors.h"
#include "heap_allocator/include/HeapCpp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <new>
#include <string>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// Deterministic temp path helper. Uses Environment::TempDirectoryPath (which
// wraps %TEMP% on Windows and $TMPDIR/"/tmp/" on Linux) plus the current
// process id to keep concurrent test runs isolated.
std::wstring
MakeTempPath(const char *stem)
{
    std::wstring path = Environment::TempDirectoryPath();
    path += L"fractalshark_test_";
    while (*stem) {
        path.push_back(static_cast<wchar_t>(*stem++));
    }
    path += L"_";
    path += std::to_wstring(Environment::CurrentProcessId());
    path += L".bin";
    return path;
}

#ifndef _WIN32
std::string
WideAsciiToNarrow(const std::wstring &path)
{
    std::string result;
    result.reserve(path.size());
    for (wchar_t ch : path) {
        result.push_back(static_cast<char>(ch));
    }
    return result;
}

uint64_t
AllocatedBytesForPath(const std::wstring &path)
{
    struct stat st{};
    const std::string narrowPath = WideAsciiToNarrow(path);
    if (stat(narrowPath.c_str(), &st) != 0) {
        return UINT64_MAX;
    }

    return static_cast<uint64_t>(st.st_blocks) * 512u;
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// Anonymous pointer stability: committing additional pages via
// MutableReserveKeepFileSize must not relocate m_Data -- the whole point of
// the reserve-then-commit design.
// ---------------------------------------------------------------------------
TEST(GrowableVector_AnonymousPointerStability)
{
    // 128 MiB virtual reserve; we never come close to this physical footprint.
    constexpr size_t ReserveBytes = 128ull * 1024ull * 1024ull;
    GrowableVector<uint32_t> vec(AddPointOptions::DontSave, L"", ReserveBytes);

    // Commit an initial slice and push a sentinel.
    vec.MutableReserveKeepFileSize(4);
    vec.PushBack(0xDEADBEEFu);

    const uint32_t *firstPtr = &vec[0];
    const uint32_t firstValue = vec[0];

    // Commit progressively larger capacities inside the same reservation. Push
    // data to make leaks observable.
    const size_t capacities[] = {128, 4096, 65536, 1ull << 20};
    for (size_t cap : capacities) {
        vec.MutableReserveKeepFileSize(cap);
        vec.PushBack(static_cast<uint32_t>(cap));
    }

    ASSERT_TRUE(&vec[0] == firstPtr);
    ASSERT_EQ(vec[0], firstValue);
    ASSERT_EQ(vec.GetSize(), static_cast<size_t>(1 + 4));
}

// ---------------------------------------------------------------------------
// Large anonymous reserve: a 1 GiB virtual reservation must succeed on
// a 64-bit host without committing the full physical footprint.
// ---------------------------------------------------------------------------
TEST(GrowableVector_LargeAnonymousReserve)
{
    constexpr size_t ReserveBytes = 1ull * 1024ull * 1024ull * 1024ull; // 1 GiB
    GrowableVector<uint8_t> vec(AddPointOptions::DontSave, L"", ReserveBytes);

    vec.MutableReserveKeepFileSize(16);
    vec.PushBack(0x42);
    vec.PushBack(0x43);

    ASSERT_EQ(vec.GetSize(), static_cast<size_t>(2));
    ASSERT_EQ(vec[0], static_cast<uint8_t>(0x42));
    ASSERT_EQ(vec[1], static_cast<uint8_t>(0x43));
}

// ---------------------------------------------------------------------------
// File-backed roundtrip: write, close, reopen. On reopen the element count
// is recovered from the on-disk file size and values read back match.
// ---------------------------------------------------------------------------
TEST(GrowableVector_FileBackedRoundtrip)
{
    const std::wstring path = MakeTempPath("roundtrip");
    Environment::FileDelete(path.c_str());

    constexpr size_t Count = 64;

    // Write phase.
    {
        GrowableVector<uint32_t> writer(AddPointOptions::EnableWithSave, path.c_str());
        writer.MutableReserveKeepFileSize(Count);
        for (uint32_t i = 0; i < Count; ++i) {
            writer.PushBack(0x1000u + i);
        }
        ASSERT_EQ(writer.GetSize(), Count);
        writer.Trim();
    } // destructor flushes + closes

    // File should exist and be exactly Count * sizeof(uint32_t) bytes.
    auto onDiskSize = Environment::FileSizeBytes(path.c_str());
    ASSERT_TRUE(onDiskSize.has_value());
    ASSERT_EQ(static_cast<size_t>(*onDiskSize), Count * sizeof(uint32_t));

    // Read phase: OpenExistingWithSave recovers the element count from size.
    {
        GrowableVector<uint32_t> reader(AddPointOptions::OpenExistingWithSave, path.c_str());
        ASSERT_EQ(reader.GetSize(), Count);
        for (uint32_t i = 0; i < Count; ++i) {
            ASSERT_EQ(reader[i], 0x1000u + i);
        }
    }

    Environment::FileDelete(path.c_str());
}

// ---------------------------------------------------------------------------
// Temporary file-backed vectors should reserve one mapping and grow capacity
// inside it without relocating m_Data.
// ---------------------------------------------------------------------------
TEST(GrowableVector_TemporaryMappedPointerStability)
{
    const std::wstring path = MakeTempPath("temporary_sparse");
    Environment::FileDelete(path.c_str());

    constexpr size_t ReserveBytes = 128ull * 1024ull * 1024ull;
    GrowableVector<uint8_t> vec(AddPointOptions::EnableWithoutSave, path.c_str(), ReserveBytes);

    vec.MutableReserveKeepFileSize(4096);
    uint8_t *firstPtr = vec.GetData();
    ASSERT_TRUE(firstPtr != nullptr);

    vec[0] = 0x5A;
    vec.MutableReserveKeepFileSize(ReserveBytes / 2);
    ASSERT_TRUE(vec.GetData() == firstPtr);

    vec[(ReserveBytes / 2) - 1] = 0xA5;
    ASSERT_EQ(vec[0], static_cast<uint8_t>(0x5A));
    ASSERT_EQ(vec[(ReserveBytes / 2) - 1], static_cast<uint8_t>(0xA5));

    ASSERT_FALSE(Environment::FileSizeBytes(path.c_str()).has_value());
}

// ---------------------------------------------------------------------------
// Linux sparse files should report a large logical size without allocating
// matching disk blocks until mapped pages are actually written.
// ---------------------------------------------------------------------------
TEST(GrowableVector_LinuxSparseFileConsumesBlocksOnlyWhenWritten)
{
#ifndef _WIN32
    const std::wstring path = MakeTempPath("sparse_blocks");
    Environment::FileDelete(path.c_str());

    constexpr size_t ReserveBytes = 128ull * 1024ull * 1024ull;

    {
        GrowableVector<uint8_t> vec(AddPointOptions::EnableWithSave, path.c_str(), ReserveBytes);
        vec.MutableResize(ReserveBytes);

        auto logicalSize = Environment::FileSizeBytes(path.c_str());
        ASSERT_TRUE(logicalSize.has_value());
        ASSERT_EQ(static_cast<size_t>(*logicalSize), ReserveBytes);

        const uint64_t allocatedBefore = AllocatedBytesForPath(path);
        ASSERT_NE(allocatedBefore, UINT64_MAX);
        ASSERT_TRUE(allocatedBefore < 1024ull * 1024ull);

        const size_t pageSize = Environment::SystemPageSize();
        vec[0] = 0x11;
        vec[ReserveBytes - pageSize] = 0x22;

        ASSERT_EQ(msync(vec.GetData(), pageSize, MS_SYNC), 0);
        ASSERT_EQ(msync(vec.GetData() + ReserveBytes - pageSize, pageSize, MS_SYNC), 0);

        const uint64_t allocatedAfter = AllocatedBytesForPath(path);
        ASSERT_NE(allocatedAfter, UINT64_MAX);
        ASSERT_TRUE(allocatedAfter >= allocatedBefore);
        ASSERT_TRUE(allocatedAfter < 16ull * 1024ull * 1024ull);
    }

    Environment::FileDelete(path.c_str());
#endif
}

// ---------------------------------------------------------------------------
// Normal test runs route process allocations through HeapCpp.
// ---------------------------------------------------------------------------
TEST(CustomHeap_MallocAndNewUseGlobalHeap)
{
    void *mallocPtr = std::malloc(257);
    ASSERT_TRUE(mallocPtr != nullptr);
    ASSERT_TRUE(GlobalHeap().OwnsPointer(mallocPtr));
    std::memset(mallocPtr, 0x31, 257);

    void *reallocPtr = std::realloc(mallocPtr, 1024);
    ASSERT_TRUE(reallocPtr != nullptr);
    ASSERT_TRUE(GlobalHeap().OwnsPointer(reallocPtr));
    std::free(reallocPtr);

    int *newPtr = new int(42);
    ASSERT_TRUE(GlobalHeap().OwnsPointer(newPtr));
    ASSERT_EQ(*newPtr, 42);
    delete newPtr;

    void *alignedPtr = ::operator new(64, std::align_val_t{16});
    ASSERT_TRUE(GlobalHeap().OwnsPointer(alignedPtr));
    ::operator delete(alignedPtr, std::align_val_t{16});
}

// ---------------------------------------------------------------------------
// The heap backing store should be a temporary file-backed GrowableVector.
// Delete-on-close parity means HeapFile.bin should not exist as a cwd entry;
// POSIX also lets us validate the live file descriptor.
// ---------------------------------------------------------------------------
TEST(CustomHeap_UsesTemporaryFileBackedGrowableVector)
{
    void *mallocPtr = std::malloc(257);
    ASSERT_TRUE(mallocPtr != nullptr);

    const auto diag = GlobalHeap().GetBackingDiagnostics();
    ASSERT_EQ(static_cast<int>(diag.AddPointOption),
              static_cast<int>(AddPointOptions::EnableWithoutSave));
    ASSERT_TRUE(diag.Filename != nullptr);
    ASSERT_EQ(std::wcscmp(diag.Filename, L"HeapFile.bin"), 0);
    ASSERT_TRUE(diag.Data != nullptr);
    ASSERT_TRUE(diag.CapacityBytes >= HEAP_INIT_SIZE);
    ASSERT_TRUE(diag.HeapBytes >= HEAP_INIT_SIZE);
    ASSERT_FALSE(Environment::FileSizeBytes(diag.Filename).has_value());

#ifndef _WIN32
    struct stat st{};
    ASSERT_EQ(fstat(static_cast<int>(diag.FileHandle), &st), 0);
    ASSERT_TRUE(S_ISREG(st.st_mode));
    ASSERT_TRUE(static_cast<uint64_t>(st.st_size) >= diag.CapacityBytes);
#endif

    std::free(mallocPtr);
}
