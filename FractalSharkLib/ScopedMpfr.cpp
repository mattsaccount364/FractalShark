#include "ScopedMpfr.h"
#include "HighPrecision.h"

thread_local uint8_t scoped_mpfr_allocators::Allocated[NumBlocks][BytesPerBlock];
thread_local size_t scoped_mpfr_allocators::AllocatedSize[NumBlocks];
thread_local size_t scoped_mpfr_allocators::AllocatedIndex = 0;
thread_local size_t scoped_mpfr_allocators::AllocationsAndFrees[NumBlocks];

void* scoped_mpfr_allocators::NewMalloc(size_t size) {
    const auto alignedSize = (size + 63) & ~63;
    if (AllocatedSize[AllocatedIndex] + alignedSize >= BytesPerBlock) {
        // Find the next block with no allocations:
        for (;;) {
            AllocatedIndex = (AllocatedIndex + 1) % NumBlocks;
            if (AllocationsAndFrees[AllocatedIndex] == 0) {
                break;
            }
        }
    }

    auto ret = &Allocated[AllocatedIndex][AllocatedSize[AllocatedIndex]];

    // Increment AllocatedSize by 64-byte aligned size:
    AllocatedSize[AllocatedIndex] += alignedSize;
    AllocationsAndFrees[AllocatedIndex]++;

    return ret;
}

void* scoped_mpfr_allocators::NewRealloc(void* ptr, size_t old_size, size_t new_size) {
    // This one is like new_malloc, but copy in the prior data.
    auto ret = NewMalloc(new_size);
    memcpy(ret, ptr, old_size);
    return ret;
}

void scoped_mpfr_allocators::NewFree(void* ptr, size_t size) {
    const auto alignedSize = (size + 63) & ~63;

    // Find offset relative to base of Allocated:
    const uint64_t offset = (uint8_t*)ptr - (uint8_t*)&Allocated[0][0];

    // Find index of Allocated:
    const size_t index = offset / BytesPerBlock;

    AllocationsAndFrees[index]--;
    AllocatedSize[index] -= alignedSize;
}

scoped_mpfr_allocators::scoped_mpfr_allocators() {
    mp_get_memory_functions(
        &ExistingMalloc,
        &ExistingRealloc,
        &ExistingFree);

    mp_set_memory_functions(
        NewMalloc,
        NewRealloc,
        NewFree);
}

scoped_mpfr_allocators::~scoped_mpfr_allocators() {
    mp_set_memory_functions(
        ExistingMalloc,
        ExistingRealloc,
        ExistingFree);
}

scoped_mpfr_precision::scoped_mpfr_precision(unsigned digits10) : saved_digits10(HighPrecision::thread_default_precision())
{
    HighPrecision::default_precision(digits10);
}
scoped_mpfr_precision::~scoped_mpfr_precision()
{
    HighPrecision::default_precision(saved_digits10);
}
void scoped_mpfr_precision::reset(unsigned digits10)
{
    HighPrecision::default_precision(digits10);
}
void scoped_mpfr_precision::reset()
{
    HighPrecision::default_precision(saved_digits10);
}

scoped_mpfr_precision_options::scoped_mpfr_precision_options(boost::multiprecision::variable_precision_options opts) : saved_options(HighPrecision::thread_default_variable_precision_options())
{
    HighPrecision::thread_default_variable_precision_options(opts);
}
scoped_mpfr_precision_options::~scoped_mpfr_precision_options()
{
    HighPrecision::thread_default_variable_precision_options(saved_options);
}
void scoped_mpfr_precision_options::reset(boost::multiprecision::variable_precision_options opts)
{
    HighPrecision::thread_default_variable_precision_options(opts);
}
