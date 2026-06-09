#include "TestFramework.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace {

bool
IsAligned(const void *ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

void
FillBytes(void *ptr, size_t size, unsigned char value)
{
    std::memset(ptr, value, size);
}

void
AssertBytes(const void *ptr, size_t size, unsigned char value)
{
    const auto *bytes = static_cast<const unsigned char *>(ptr);
    for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(bytes[i], value);
    }
}

} // namespace

TEST(HeapMallocIsAtLeast64ByteAligned)
{
    void *ptr = std::malloc(123);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, 64));
    std::free(ptr);
}

TEST(HeapAlignedAllocSupportsLargeNonMultipleSize)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t Size = 524336;

#ifdef _MSC_VER
    void *ptr = _aligned_malloc(Size, Alignment);
#else
    void *ptr = aligned_alloc(Alignment, Size);
#endif

    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    FillBytes(ptr, Size, 0x5A);
    AssertBytes(ptr, Size, 0x5A);

#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

TEST(HeapAlignedNewSupportsLargeAlignment)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t Size = 524336;

    void *ptr = ::operator new(Size, std::align_val_t{Alignment});
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    FillBytes(ptr, Size, 0xA5);
    AssertBytes(ptr, Size, 0xA5);
    ::operator delete(ptr, std::align_val_t{Alignment});
}

TEST(HeapMultipleOverAlignedAllocationsFreeInMixedOrder)
{
    void *p128 = nullptr;
    void *p256 = nullptr;
    void *p4096 = nullptr;
    void *p65536 = nullptr;

#ifdef _MSC_VER
    p128 = _aligned_malloc(257, 128);
    p256 = _aligned_malloc(4097, 256);
    p4096 = _aligned_malloc(524336, 4096);
    p65536 = _aligned_malloc(777, 65536);
#else
    ASSERT_EQ(posix_memalign(&p128, 128, 257), 0);
    ASSERT_EQ(posix_memalign(&p256, 256, 4097), 0);
    ASSERT_EQ(posix_memalign(&p4096, 4096, 524336), 0);
    ASSERT_EQ(posix_memalign(&p65536, 65536, 777), 0);
#endif

    ASSERT_TRUE(IsAligned(p128, 128));
    ASSERT_TRUE(IsAligned(p256, 256));
    ASSERT_TRUE(IsAligned(p4096, 4096));
    ASSERT_TRUE(IsAligned(p65536, 65536));

#ifdef _MSC_VER
    _aligned_free(p4096);
    _aligned_free(p128);
    _aligned_free(p65536);
    _aligned_free(p256);
#else
    std::free(p4096);
    std::free(p128);
    std::free(p65536);
    std::free(p256);
#endif
}

TEST(HeapReallocCopiesOverAlignedAllocation)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t OldSize = 4093;
    constexpr size_t NewSize = 9000;

#ifdef _MSC_VER
    void *ptr = _aligned_malloc(OldSize, Alignment);
#else
    void *ptr = nullptr;
    ASSERT_EQ(posix_memalign(&ptr, Alignment, OldSize), 0);
#endif

    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    FillBytes(ptr, OldSize, 0xC3);

#ifdef _MSC_VER
    void *newPtr = _aligned_realloc(ptr, NewSize, Alignment);
#else
    void *newPtr = std::realloc(ptr, NewSize);
#endif

    ASSERT_TRUE(newPtr != nullptr);
#ifdef _MSC_VER
    ASSERT_TRUE(IsAligned(newPtr, Alignment));
#endif
    AssertBytes(newPtr, OldSize, 0xC3);

#ifdef _MSC_VER
    _aligned_free(newPtr);
#else
    std::free(newPtr);
#endif
}
