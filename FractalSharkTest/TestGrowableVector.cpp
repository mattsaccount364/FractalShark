#include "TestFramework.h"

#include "Environment.h"
#include "Vectors.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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

} // namespace

// ---------------------------------------------------------------------------
// Anonymous pointer stability: committing additional pages via
// MutableReserveKeepFileSize must not relocate m_Data — the whole point of
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

    // Commit progressively larger capacities — each call mprotects more pages
    // within the same reservation. Push data to make leaks observable.
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
// Large anonymous reserve: a 1 GiB PROT_NONE reservation must succeed on
// a 64-bit host (Linux overcommit plus PROT_NONE keep this nearly free).
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
// is recovered from the on-disk file size (fstat) and values read back match.
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

    // Read phase — OpenExistingWithSave recovers the element count from size.
    {
        GrowableVector<uint32_t> reader(AddPointOptions::OpenExistingWithSave, path.c_str());
        ASSERT_EQ(reader.GetSize(), Count);
        for (uint32_t i = 0; i < Count; ++i) {
            ASSERT_EQ(reader[i], 0x1000u + i);
        }
    }

    Environment::FileDelete(path.c_str());
}
