#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

struct ScopedMPIRAllocators {
    void* (*ExistingMalloc) (size_t);
    void* (*ExistingRealloc) (void*, size_t, size_t);
    void (*ExistingFree) (void*, size_t);

    ScopedMPIRAllocators();
    ~ScopedMPIRAllocators();

    static constexpr size_t NumBlocks = 4;
    static constexpr size_t BytesPerBlock = 1024 * 1024;

    thread_local static uint8_t Allocated[NumBlocks][BytesPerBlock];
    thread_local static size_t AllocatedSize[NumBlocks];
    thread_local static size_t AllocatedIndex;
    thread_local static size_t AllocationsAndFrees[NumBlocks];

private:
    static void* NewMalloc(size_t size);
    static void* NewRealloc(void* ptr, size_t old_size, size_t new_size);
    static void NewFree(void* ptr, size_t size);
};

struct ScopedMPIRPrecision
{
    unsigned saved_digits10;
    ScopedMPIRPrecision(unsigned digits10);
    ~ScopedMPIRPrecision();
    void reset(unsigned digits10);
    void reset();
};

struct ScopedMPIRPrecisionOptions
{
    boost::multiprecision::variable_precision_options saved_options;
    ScopedMPIRPrecisionOptions(boost::multiprecision::variable_precision_options opts);
    ~ScopedMPIRPrecisionOptions();
    void reset(boost::multiprecision::variable_precision_options opts);
};
