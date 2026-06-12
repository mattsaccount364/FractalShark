#include "TestFramework.h"

#include "heap_allocator/include/HeapCpp.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _MSC_VER
#include <malloc.h>
#endif

extern "C" void *aligned_alloc(size_t alignment, size_t size);

void *CppAlignedMalloc(size_t size, size_t alignment);
void *CppAlignedRealloc(void *ptr, size_t newUserSize, size_t alignment, bool zeroNew);

namespace {

bool
IsAligned(const void *ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

void
AssertHeapOwned(const void *ptr)
{
    ASSERT_TRUE(GlobalHeap().OwnsPointer(ptr));
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

TEST(HeapCAllocationApisUseGlobalHeap)
{
    void *ptr = std::malloc(123);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, 64));
    AssertHeapOwned(ptr);
    FillBytes(ptr, 123, 0x33);

    void *reallocPtr = std::realloc(ptr, 512);
    ASSERT_TRUE(reallocPtr != nullptr);
    ASSERT_TRUE(IsAligned(reallocPtr, 64));
    AssertHeapOwned(reallocPtr);
    AssertBytes(reallocPtr, 123, 0x33);
    std::free(reallocPtr);

    auto *callocPtr = static_cast<unsigned char *>(std::calloc(17, 11));
    ASSERT_TRUE(callocPtr != nullptr);
    ASSERT_TRUE(IsAligned(callocPtr, 64));
    AssertHeapOwned(callocPtr);
    AssertBytes(callocPtr, 17 * 11, 0);
    std::free(callocPtr);
}

TEST(HeapAlignedAllocSupportsLargeNonMultipleSize)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t Size = 524336;

    void *ptr = aligned_alloc(Alignment, Size);

    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    AssertHeapOwned(ptr);
    FillBytes(ptr, Size, 0x5A);
    AssertBytes(ptr, Size, 0x5A);

    std::free(ptr);
}

TEST(HeapAlignedNewSupportsLargeAlignment)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t Size = 524336;

    void *ptr = ::operator new(Size, std::align_val_t{Alignment});
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    AssertHeapOwned(ptr);
    FillBytes(ptr, Size, 0xA5);
    AssertBytes(ptr, Size, 0xA5);
    ::operator delete(ptr, std::align_val_t{Alignment});
}

TEST(HeapMultipleOverAlignedAllocationsFreeInMixedOrder)
{
    void *p128 = aligned_alloc(128, 257);
    void *p256 = aligned_alloc(256, 4097);
    void *p4096 = aligned_alloc(4096, 524336);
    void *p65536 = aligned_alloc(65536, 777);
    void *normal = std::malloc(333);

    ASSERT_TRUE(p128 != nullptr);
    ASSERT_TRUE(p256 != nullptr);
    ASSERT_TRUE(p4096 != nullptr);
    ASSERT_TRUE(p65536 != nullptr);
    ASSERT_TRUE(normal != nullptr);
    ASSERT_TRUE(IsAligned(p128, 128));
    ASSERT_TRUE(IsAligned(p256, 256));
    ASSERT_TRUE(IsAligned(p4096, 4096));
    ASSERT_TRUE(IsAligned(p65536, 65536));
    ASSERT_TRUE(IsAligned(normal, 64));
    AssertHeapOwned(p128);
    AssertHeapOwned(p256);
    AssertHeapOwned(p4096);
    AssertHeapOwned(p65536);
    AssertHeapOwned(normal);

    std::free(p4096);
    std::free(normal);
    std::free(p128);
    std::free(p65536);
    std::free(p256);
}

TEST(HeapReallocCopiesOverAlignedAllocation)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t OldSize = 4093;
    constexpr size_t NewSize = 9000;

    void *ptr = CppAlignedMalloc(OldSize, Alignment);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    AssertHeapOwned(ptr);
    FillBytes(ptr, OldSize, 0xC3);

    void *newPtr = CppAlignedRealloc(ptr, NewSize, Alignment, false);
    ASSERT_TRUE(newPtr != nullptr);
    ASSERT_TRUE(IsAligned(newPtr, Alignment));
    AssertHeapOwned(newPtr);
    AssertBytes(newPtr, OldSize, 0xC3);

    std::free(newPtr);
}

#ifdef _MSC_VER
TEST(HeapWindowsAlignedAllocCompatibility)
{
    constexpr size_t Alignment = 4096;
    constexpr size_t Size = 524336;

    void *ptr = _aligned_malloc(Size, Alignment);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, Alignment));
    AssertHeapOwned(ptr);
    ASSERT_TRUE(_aligned_msize(ptr, Alignment, 0) >= Size);
    FillBytes(ptr, Size, 0x81);
    AssertBytes(ptr, Size, 0x81);
    _aligned_free(ptr);
}
#else
TEST(HeapPosixMemalignCompatibility)
{
    void *ptr = nullptr;
    ASSERT_EQ(posix_memalign(&ptr, 4096, 4093), 0);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_TRUE(IsAligned(ptr, 4096));
    AssertHeapOwned(ptr);
    FillBytes(ptr, 4093, 0x81);
    AssertBytes(ptr, 4093, 0x81);
    std::free(ptr);
}
#endif
